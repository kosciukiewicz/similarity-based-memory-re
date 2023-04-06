from copy import deepcopy

from transformers import PreTrainedTokenizer


class RelationType:
    def __init__(
        self,
        identifier: str,
        index: int,
        short_name: str | None = None,
        verbose_name: str | None = None,
        symmetric: bool = False,
    ):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name if short_name else identifier
        self._verbose_name = verbose_name if verbose_name else identifier
        self._symmetric = symmetric

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    @property
    def symmetric(self):
        return self._symmetric

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class EntityType:
    def __init__(
        self,
        identifier: str,
        index: int,
        short_name: str | None = None,
        verbose_name: str | None = None,
    ):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name if short_name else identifier
        self._verbose_name = verbose_name if verbose_name else identifier

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class Token:
    def __init__(
        self, tid: int, doc_index: int, sent_index: int, span_start: int, span_end: int, phrase: str
    ):
        self._tid = tid  # ID within the corresponding dataset
        self._doc_index = doc_index  # original token index in document
        self._sent_index = sent_index  # original token index in sentence

        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end  # end of token span in document (exclusive)
        self._phrase = phrase

    @property
    def doc_index(self):
        return self._doc_index

    @property
    def sent_index(self):
        return self._sent_index

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase


class TokenSpan:
    def __init__(self, tokens: list[Token]):
        self._tokens = tokens

    @property
    def tokens(self):
        return self._tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def orig_span_start(self):
        return self._tokens[0].doc_index

    @property
    def orig_span_end(self):
        return self._tokens[-1].doc_index + 1

    @property
    def orig_span(self):
        return self.orig_span_start, self.orig_span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start : s.stop : s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


class Entity:
    def __init__(self, eid: int, doc_entity_idx: int, entity_type: EntityType, phrase: str):
        self._eid = eid
        self._doc_entity_idx = doc_entity_idx
        self._entity_type = entity_type
        self._phrase = phrase

        self._entity_mentions: list['EntityMention'] = []

    def add_entity_mention(self, mention):
        self._entity_mentions.append(mention)

    @property
    def doc_entity_index(self):
        return self._doc_entity_idx

    @property
    def entity_type(self) -> EntityType:
        return self._entity_type

    @property
    def entity_mentions(self) -> list['EntityMention']:
        return self._entity_mentions

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase


class EntityMention:
    def __init__(
        self, emid: int, entity: Entity, tokens: list[Token], sentence: 'Sentence', phrase: str
    ):
        self._emid = emid  # ID within the corresponding dataset

        self._entity = entity

        self._tokens = tokens
        self._sentence = sentence
        self._phrase = phrase

    @property
    def entity(self):
        return self._entity

    @property
    def entity_type(self) -> EntityType:
        return self._entity.entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def orig_span_start(self):
        return self._tokens[0].doc_index

    @property
    def orig_span_end(self):
        return self._tokens[-1].doc_index + 1

    @property
    def orig_span(self):
        return self.orig_span_start, self.orig_span_end

    @property
    def sentence(self):
        return self._sentence

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, EntityMention):
            return self._emid == other._emid
        return False

    def __hash__(self):
        return hash(self._emid)

    def __str__(self):
        return self._phrase


class Sentence:
    def __init__(self, sent_id: int, index: int, tokens: list[Token]):
        self._sent_id = sent_id  # ID within the corresponding dataset
        self._index = index  # ID within the corresponding document
        self._tokens = tokens
        self._entity_mentions: list['EntityMention'] = []

    def add_entity_mention(self, entity_mention):
        self._entity_mentions.append(entity_mention)

    @property
    def sent_id(self):
        return self._sent_id

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def entity_mentions(self):
        return self._entity_mentions

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    def __str__(self):
        return ' '.join([str(t) for t in self.tokens])

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Sentence):
            return self._sent_id == other._sent_id
        return False

    def __hash__(self):
        return hash(self._sent_id)


class Relation:
    def __init__(
        self,
        rid: int,
        relation_type: RelationType,
        head_entity: Entity,
        tail_entity: Entity,
        evidence_sentences: list[Sentence],
    ):
        self._rid = rid  # ID within the corresponding dataset
        self._relation_type = relation_type

        self._head_entity = head_entity
        self._tail_entity = tail_entity

        self._evidence_sentences = evidence_sentences

    @property
    def relation_type(self) -> RelationType:
        return self._relation_type

    @property
    def head_entity(self) -> Entity:
        return self._head_entity

    @property
    def tail_entity(self) -> Entity:
        return self._tail_entity

    @property
    def evidence_sentences(self):
        return self._evidence_sentences

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self._rid == other._rid
        return False

    def __hash__(self):
        return hash(self._rid)


class Document:
    def __init__(
        self,
        doc_id: int,
        doc_source_file: str,
        doc_source_id: int,
        tokens: list[Token],
        sentences: list[Sentence],
        entities: list[Entity],
        relations: list[Relation],
        title: str,
    ):
        self._doc_id = doc_id  # ID within the corresponding dataset
        self._doc_source_file = doc_source_file
        self._doc_source_id = doc_source_id
        self._sentences = sentences
        self._tokens = tokens
        self._entities = entities
        self._relations = relations
        self._entity_mentions = [m for e in entities for m in e.entity_mentions]

        self._title = title

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def sentences(self):
        return self._sentences

    @property
    def entities(self):
        return self._entities

    @property
    def entity_mentions(self):
        return self._entity_mentions

    @property
    def relations(self):
        return self._relations

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def title(self):
        return self._title

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ' '.join([str(s) for s in self.sentences])

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)


class TokenizedDocument(Document):
    _UNKNOWN_TOKEN = '[UNK]'
    _CLS_TOKEN = '[CLS]'
    _SEP_TOKEN = '[SEP]'

    def __init__(
        self,
        doc_id: int,
        doc_source_file: str,
        doc_source_id: int,
        tokens: list[Token],
        sentences: list[Sentence],
        entities: list[Entity],
        relations: list[Relation],
        encoding: list[int],
        title: str,
    ):
        super().__init__(
            doc_id, doc_source_file, doc_source_id, tokens, sentences, entities, relations, title
        )

        self._encoding = encoding

    @classmethod
    def from_document(cls, doc: Document, tokenizer: PreTrainedTokenizer) -> 'TokenizedDocument':
        source_doc = deepcopy(doc)

        doc_encoding: list[int] = []  # type: ignore
        for sentence in source_doc.sentences:
            for token in sentence.tokens:
                token_encoding = tokenizer.encode(token.phrase, add_special_tokens=False)
                if not token_encoding:
                    token_encoding = [
                        tokenizer.convert_tokens_to_ids(cls._UNKNOWN_TOKEN)
                    ]  # type: ignore

                token._span_start = len(doc_encoding)
                token._span_end = len(doc_encoding) + len(token_encoding)
                doc_encoding += token_encoding

        return TokenizedDocument(
            doc_id=source_doc.doc_id,
            doc_source_file=source_doc._doc_source_file,
            doc_source_id=source_doc._doc_source_id,
            tokens=source_doc._tokens,
            sentences=source_doc.sentences,
            entities=source_doc.entities,
            relations=source_doc.relations,
            encoding=doc_encoding,
            title=source_doc.title,
        )

    @property
    def encodings(self):
        return self._encoding
