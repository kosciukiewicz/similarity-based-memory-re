import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from memory_re.data.datasets.entities import (
    Document,
    Entity,
    EntityMention,
    EntityType,
    Relation,
    RelationType,
    Sentence,
    Token,
    TokenizedDocument,
)
from memory_re.utils.collections import flatten


class RelationExtractionDataset(Dataset):
    # OVERRIDDEN BY INHERITED CLASSES
    _BASE_DATASET_DIR: Path
    _SPLIT_FILEPATH_MAPPING: dict[str, Path | list[Path]]

    NONE_LABEL = 'None'

    def __init__(
        self,
        split: str,
        entity_types: dict[str, EntityType],
        relation_types: dict[str, RelationType],
    ):
        self._split = split
        self._entity_types = entity_types
        self._relation_types = relation_types
        # current ids
        self._doc_id = 0
        self._sid = 0
        self._rid = 0
        self._eid = 0
        self._meid = 0
        self._tid = 0

        self._documents: list[Document | TokenizedDocument] = []
        self._entity_mentions: OrderedDict[int, EntityMention] = OrderedDict()
        self._entities: OrderedDict[int, Entity] = OrderedDict()
        self._relations: OrderedDict[int, Relation] = OrderedDict()
        self._is_data_loaded = False
        self._is_tokenized = False

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index) -> Document:
        return self._documents[index]

    @property
    def is_data_loaded(self) -> bool:
        return self._is_data_loaded

    @classmethod
    def types_filepath(cls) -> Path:
        return cls._BASE_DATASET_DIR / 'types.json'

    @property
    def documents(self) -> list[Document]:
        return self._documents

    def tokenize(self, tokenizer: PreTrainedTokenizer) -> None:
        if self._is_tokenized:
            return

        new_documents: list[Document | TokenizedDocument] = []
        desc = "Tokenize dataset '{}' for '{}' split".format(self.__class__.__name__, self._split)

        for doc in tqdm(self._documents, desc=desc):
            new_documents.append(TokenizedDocument.from_document(doc, tokenizer))
        self._documents = new_documents
        self._is_tokenized = True

    def load_documents(self) -> None:
        if not self._is_data_loaded:
            dataset_filepaths = self._SPLIT_FILEPATH_MAPPING[self._split]
            if not isinstance(dataset_filepaths, list):
                dataset_filepaths = [dataset_filepaths]

            for dataset_filepath in dataset_filepaths:
                raw_documents = json.load(open(dataset_filepath))
                desc = "Parse dataset '%s'" % dataset_filepath
                for i, raw_document in enumerate(tqdm(raw_documents, desc=desc)):
                    self._parse_document(
                        doc_source_filepath=dataset_filepath,
                        raw_doc_id=i,
                        raw_document=raw_document,
                    )

            self._is_data_loaded = True

    def _parse_document(
        self, doc_source_filepath: Path, raw_doc_id: int, raw_document: dict[str, Any]
    ) -> None:
        title = raw_document.get('title', '')
        sentences = self._parse_sentences(raw_document['sents'])
        entities = self._parse_entities(raw_document['entities'], sentences)
        relations = self._parse_relations(raw_document.get('relations', []), entities, sentences)
        doc_tokens = flatten([s.tokens for s in sentences])

        document = Document(
            doc_id=self._doc_id,
            doc_source_id=raw_doc_id,
            doc_source_file=doc_source_filepath.stem,
            tokens=doc_tokens,
            entities=entities,
            relations=relations,
            sentences=sentences,
            title=title,
        )
        self._documents.append(document)
        self._doc_id += 1

    def _parse_relations(
        self, raw_relations: list[dict[str, Any]], entities: list[Entity], sentences: list[Sentence]
    ) -> list[Relation]:
        relations = []

        for raw_relation in raw_relations:
            relation_type = self._relation_types[raw_relation['type']]

            head_idx = raw_relation['head']
            tail_idx = raw_relation['tail']

            evidence = raw_relation['evidence_sent_id']
            evidence_sentences = [sentences[ev] for ev in evidence]

            # create relation
            head_entity = entities[head_idx]
            tail_entity = entities[tail_idx]

            relation = self._create_relation(
                relation_type, head_entity, tail_entity, evidence_sentences
            )
            relations.append(relation)

        return relations

    def _parse_entities(
        self, raw_entities: list[list[dict[str, Any]]], sentences: list[Sentence]
    ) -> list[Entity]:
        entities = []

        for entity_idx, raw_entity in enumerate(raw_entities):
            mention_params = []
            for raw_entity_mention in raw_entity:
                entity_type = self._entity_types[raw_entity_mention['type']]
                start, end = raw_entity_mention['start'], raw_entity_mention['end']

                # create entity mention
                sentence = sentences[raw_entity_mention['sent_id']]
                tokens = sentence.tokens[start:end].tokens
                phrase = " ".join([t.phrase for t in tokens])

                mention_params.append((entity_type, tokens, phrase, sentence))

            entity_type = mention_params[0][0]
            entity_phrase = mention_params[0][2]
            entity = self._create_entity(entity_type, entity_phrase, entity_idx)

            for _, tokens, phrase, sentence in mention_params:
                entity_mention = self._create_entity_mention(entity, tokens, sentence, phrase)

                entity.add_entity_mention(entity_mention)
                sentence.add_entity_mention(entity_mention)

            entities.append(entity)

        return entities

    def _parse_sentences(self, raw_sentences: dict[str, Any]) -> list[Sentence]:
        sentences = []

        tok_doc_idx = 0

        current_doc_length = 0

        for s_i, raw_sentence_tokens in enumerate(raw_sentences):
            sentence_tokens = []
            for token_phrase in raw_sentence_tokens:
                span_start, span_end = (current_doc_length, current_doc_length + 1)
                token = self._create_token(tok_doc_idx, s_i, span_start, span_end, token_phrase)

                sentence_tokens.append(token)

                tok_doc_idx += 1
                current_doc_length += 1

            sentence = self._create_sentence(s_i, sentence_tokens)
            sentences.append(sentence)

        return sentences

    def _create_relation(
        self,
        relation_type: RelationType,
        head_entity: Entity,
        tail_entity: Entity,
        evidence_sentences: list[Sentence],
    ) -> Relation:
        relation = Relation(self._rid, relation_type, head_entity, tail_entity, evidence_sentences)
        self._relations[self._rid] = relation
        self._rid += 1
        return relation

    def _create_entity(self, entity_type: EntityType, phrase: str, doc_entity_idx: int) -> Entity:
        entity = Entity(self._eid, doc_entity_idx, entity_type, phrase)
        self._entities[self._eid] = entity
        self._eid += 1
        return entity

    def _create_entity_mention(
        self, entity: Entity, tokens: list[Token], sentence: Sentence, phrase: str
    ) -> EntityMention:
        mention = EntityMention(self._meid, entity, tokens, sentence, phrase)
        self._entity_mentions[self._meid] = mention
        self._meid += 1
        return mention

    def _create_sentence(self, index: int, tokens: list[Token]) -> Sentence:
        sentence = Sentence(self._sid, index, tokens)
        self._sid += 1
        return sentence

    def _create_token(
        self, doc_index: int, sent_index: int, span_start: int, span_end: int, phrase: str
    ) -> Token:
        token = Token(self._tid, doc_index, sent_index, span_start, span_end, phrase)
        self._tid += 1
        return token
