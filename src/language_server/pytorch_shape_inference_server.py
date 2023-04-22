import ast
import re
from typing import List, Tuple
from pygls.server import LanguageServer
from lsprotocol import types


class TensorAnalyazerServer(LanguageServer):
    def __init__(self, name="tensor-analyzer", version="0.0.1"):
        super().__init__(name, version)
        self.shape_inference_in_progress = False

server = TensorAnalyazerServer()

def infer_shapes(text: str, uri: str) -> Tuple[List[types.TextEdit], bool]:
    text_edits = []
    changes_made = False

    class ShapeInferenceVisitor(ast.NodeVisitor):
        def __init__(self):
            self.linear_shapes = {}

        def visit_Call(self, node: ast.Call):
            nonlocal changes_made
            shape_str = None

            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "randn"
            ):
                shape = [str(arg.n) for arg in node.args]
                shape_str = f" # -> [{', '.join(shape)}]"

            elif (
                isinstance(node.func, ast.Name)
                and isinstance(node.func.id, str)
                and node.func.id in self.linear_shapes
            ):
                in_shape, out_shape = self.linear_shapes[node.func.id]
                shape_str = f" # -> {out_shape}"

            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "nn"
                and node.func.attr == "Linear"
            ):
                in_features, out_features = [str(arg.n) for arg in node.args]
                shape_str = f" # -> [_, {in_features}] -> [_, {out_features}]"
                self.linear_shapes[node] = (f"[_ {in_features}]", f"[_ {out_features}]")

            if shape_str is not None:
                text_start = node.end_col_offset

                # Get the line and remove any existing shape information
                line = text.splitlines()[node.lineno - 1]
                updated_line, n = re.subn(r" # -> .*", shape_str, line)
                if n == 0:  # If no existing shape information was replaced, append the new shape_str
                    updated_line += shape_str

                # Create a text edit for the updated line
                text_edit = types.TextEdit(
                    range=types.Range(
                        start=types.Position(line=node.lineno - 1, character=0),
                        end=types.Position(line=node.lineno - 1, character=len(line)),
                    ),
                    new_text=updated_line,
                )
                text_edits.append(text_edit)
                changes_made = True

            self.generic_visit(node)


    tree = ast.parse(text)
    visitor = ShapeInferenceVisitor()
    visitor.visit(tree)

    return text_edits, changes_made

@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(server: TensorAnalyazerServer, params: types.DidSaveTextDocumentParams) -> None:
    if server.shape_inference_in_progress:
        return

    document = server.workspace.get_document(params.text_document.uri)
    text = document.source
    text_edits, changes_made = infer_shapes(text, params.text_document.uri)
    if changes_made:
        server.shape_inference_in_progress = True
        server.apply_edit(
            types.WorkspaceEdit(changes={params.text_document.uri: text_edits})
        )
        server.shape_inference_in_progress = False


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(server: TensorAnalyazerServer, params: types.DidChangeTextDocumentParams) -> None:
    if server.shape_inference_in_progress:
        return

    document = server.workspace.get_document(params.text_document.uri)
    text = document.source
    text_edits, changes_made = infer_shapes(text, params.text_document.uri)
    if changes_made:
        server.shape_inference_in_progress = True
        server.apply_edit(
            types.WorkspaceEdit(changes={params.text_document.uri: text_edits})
        )
        server.shape_inference_in_progress = False



if __name__ == '__main__':
    server.start_io()
