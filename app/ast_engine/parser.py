"""Tree-sitter based Python symbol extractor.

Extracts, per module:
  * symbol definitions (functions, methods, classes) with exact line ranges and parameter
    signatures,
  * call sites (with positional/keyword argument shapes, used for contract checking),
  * function references passed as arguments (e.g. FastAPI's `Depends(get_current_user)`),
  * import bindings (`import x as y`, `from x import y as z`).

Tree-sitter is used instead of `ast` because it is error tolerant: a PR that leaves a file
temporarily unparsable still yields a best-effort symbol table instead of a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from app.schemas import Param

MODULE_SCOPE = "<module>"


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tspython.language()))


def parse_tree(source: str):
    return _parser().parse(source.encode("utf-8"))


@dataclass
class SymbolDef:
    module: str
    qualname: str
    name: str
    kind: str  # function | method | class
    file_path: str
    start_line: int
    end_line: int
    params: list[Param] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def fqname(self) -> str:
        return f"{self.module}.{self.qualname}"

    @property
    def signature(self) -> str:
        return render_signature(self.name, self.params)

    def contains(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line


@dataclass
class CallSite:
    caller: str  # qualname inside the module, or MODULE_SCOPE
    callee_expr: str  # e.g. `verify_token`, `auth.verify_token`, `self._sign`
    line: int
    positional_args: int
    keyword_args: list[str]
    has_star_args: bool
    has_star_kwargs: bool
    text: str
    referenced_names: list[str] = field(default_factory=list)


@dataclass
class ImportRef:
    module: str
    name: str | None  # None for `import module`
    alias: str
    line: int


@dataclass
class ParsedModule:
    module: str
    file_path: str
    source: str
    symbols: list[SymbolDef] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    imports: list[ImportRef] = field(default_factory=list)
    has_errors: bool = False

    def symbol(self, qualname: str) -> SymbolDef | None:
        return next((s for s in self.symbols if s.qualname == qualname), None)

    def innermost_symbol(self, line: int) -> SymbolDef | None:
        hits = [s for s in self.symbols if s.contains(line)]
        return min(hits, key=lambda s: s.end_line - s.start_line, default=None)

    def lines(self, start: int, end: int) -> str:
        return "\n".join(self.source.splitlines()[start - 1 : end])


def render_signature(name: str, params: list[Param]) -> str:
    parts: list[str] = []
    star_emitted = False
    for p in params:
        if p.kind == "var_positional":
            star_emitted = True
        elif p.kind == "keyword_only" and not star_emitted:
            parts.append("*")
            star_emitted = True
        parts.append(p.render())
    return f"{name}({', '.join(parts)})"


def module_name_for(rel_path: str) -> str:
    parts = list(Path(rel_path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or "__init__"


def _text(node: Node | None) -> str:
    return node.text.decode("utf-8") if node is not None and node.text is not None else ""


def _parse_params(params_node: Node | None) -> list[Param]:
    if params_node is None:
        return []
    result: list[Param] = []
    keyword_only = False
    for child in params_node.named_children:
        t = child.type
        if t == "keyword_separator":
            keyword_only = True
            continue
        if t == "positional_separator":
            continue
        kind = "keyword_only" if keyword_only else "positional"
        name, annotation, default = "", None, None
        if t == "identifier":
            name = _text(child)
        elif t in ("typed_parameter",):
            inner = child.named_children[0] if child.named_children else None
            annotation = _text(child.child_by_field_name("type")) or None
            if inner is not None and inner.type == "list_splat_pattern":
                name, kind = _text(inner).lstrip("*"), "var_positional"
                keyword_only = True
            elif inner is not None and inner.type == "dictionary_splat_pattern":
                name, kind = _text(inner).lstrip("*"), "var_keyword"
            else:
                name = _text(inner)
        elif t in ("default_parameter", "typed_default_parameter"):
            name = _text(child.child_by_field_name("name"))
            annotation = _text(child.child_by_field_name("type")) or None
            default = _text(child.child_by_field_name("value"))
        elif t == "list_splat_pattern":
            name, kind = _text(child).lstrip("*"), "var_positional"
            keyword_only = True
        elif t == "dictionary_splat_pattern":
            name, kind = _text(child).lstrip("*"), "var_keyword"
        else:
            continue
        result.append(
            Param(
                name=name,
                kind=kind,  # type: ignore[arg-type]
                has_default=default is not None,
                default=default,
                annotation=annotation,
            )
        )
    return result


def _call_shape(args: Node | None) -> tuple[int, list[str], bool, bool, list[str]]:
    positional, keywords, star, dstar, refs = 0, [], False, False, []
    if args is None:
        return positional, keywords, star, dstar, refs
    if args.type == "generator_expression":
        return 1, keywords, star, dstar, refs
    for child in args.named_children:
        if child.type == "keyword_argument":
            keywords.append(_text(child.child_by_field_name("name")))
            value = child.child_by_field_name("value")
            if value is not None and value.type in ("identifier", "attribute"):
                refs.append(_text(value))
        elif child.type == "list_splat":
            star = True
        elif child.type == "dictionary_splat":
            dstar = True
        elif child.type == "comment":
            continue
        else:
            positional += 1
            if child.type in ("identifier", "attribute"):
                refs.append(_text(child))
    return positional, keywords, star, dstar, refs


class _Visitor:
    def __init__(self, parsed: ParsedModule) -> None:
        self.m = parsed
        self.scope: list[tuple[str, str]] = []  # (name, kind)

    def _qual(self, name: str) -> str:
        return ".".join([s for s, _ in self.scope] + [name])

    def _current(self) -> str:
        return ".".join(s for s, _ in self.scope) or MODULE_SCOPE

    def visit(self, node: Node, decorators: list[str] | None = None) -> None:
        t = node.type
        if t == "decorated_definition":
            decos = [
                _text(d).lstrip("@").strip() for d in node.named_children if d.type == "decorator"
            ]
            for d in node.named_children:
                if d.type == "decorator":
                    self.visit(d)
            definition = node.child_by_field_name("definition")
            if definition is not None:
                self.visit(definition, decos)
            return
        if t in ("function_definition", "class_definition"):
            name = _text(node.child_by_field_name("name"))
            in_class = bool(self.scope) and self.scope[-1][1] == "class"
            kind = "class" if t == "class_definition" else ("method" if in_class else "function")
            self.m.symbols.append(
                SymbolDef(
                    module=self.m.module,
                    qualname=self._qual(name),
                    name=name,
                    kind=kind,
                    file_path=self.m.file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    params=_parse_params(node.child_by_field_name("parameters"))
                    if t == "function_definition"
                    else [],
                    decorators=decorators or [],
                    source=_text(node),
                )
            )
            self.scope.append((name, "class" if t == "class_definition" else "function"))
            for child in node.named_children:
                if child.type != "identifier":
                    self.visit(child)
            self.scope.pop()
            return
        if t == "call":
            fn = node.child_by_field_name("function")
            pos, kws, star, dstar, refs = _call_shape(node.child_by_field_name("arguments"))
            self.m.calls.append(
                CallSite(
                    caller=self._current(),
                    callee_expr=_text(fn),
                    line=node.start_point[0] + 1,
                    positional_args=pos,
                    keyword_args=kws,
                    has_star_args=star,
                    has_star_kwargs=dstar,
                    text=" ".join(_text(node).split())[:240],
                    referenced_names=refs,
                )
            )
        elif t == "import_statement":
            for child in node.named_children:
                if child.type == "dotted_name":
                    mod = _text(child)
                    self.m.imports.append(
                        ImportRef(mod, None, mod.split(".")[0], node.start_point[0] + 1)
                    )
                elif child.type == "aliased_import":
                    mod = _text(child.child_by_field_name("name"))
                    alias = _text(child.child_by_field_name("alias"))
                    self.m.imports.append(ImportRef(mod, None, alias, node.start_point[0] + 1))
            return
        elif t == "import_from_statement":
            mod_node = node.child_by_field_name("module_name")
            mod = _text(mod_node).lstrip(".")
            for child in node.children_by_field_name("name"):
                if child.type == "aliased_import":
                    name = _text(child.child_by_field_name("name"))
                    alias = _text(child.child_by_field_name("alias"))
                else:
                    name = alias = _text(child)
                self.m.imports.append(ImportRef(mod, name, alias, node.start_point[0] + 1))
            return
        elif t == "ERROR":
            self.m.has_errors = True
        for child in node.named_children:
            self.visit(child)


def parse_source(source: str, module: str, file_path: str) -> ParsedModule:
    parsed = ParsedModule(module=module, file_path=file_path, source=source)
    tree = parse_tree(source)
    parsed.has_errors = tree.root_node.has_error
    _Visitor(parsed).visit(tree.root_node)
    return parsed


def parse_file(path: Path, root: Path) -> ParsedModule:
    rel = path.relative_to(root).as_posix()
    return parse_source(
        path.read_text(encoding="utf-8", errors="replace"), module_name_for(rel), rel
    )
