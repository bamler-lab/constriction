import sys
from types import ModuleType
import pdoc
import pdoc.cli
import constriction

constriction.__all__ = constriction.constriction.__all__
del constriction.constriction

context = pdoc.Context()

pdoc.link_inheritance(context)

def pdocify(mod, prefix=''):
    pdoc_mod = pdoc.Module(mod)
    prefix = prefix + pdoc_mod.name + '.'

    # Remove documentation of deprecated methods.
    for child_name in pdoc_mod.doc:
        child = pdoc_mod.doc[child_name]
        try:
            grandchild_names = list(child.doc.keys()) # so we can modify the dictionary while iterating over it.
        except:
            continue
        for grandchild_name in grandchild_names:
            grandchild = child.doc[grandchild_name]
            grandchild_doc = grandchild.docstring
            if grandchild_doc is not None and grandchild_doc[:16] == '.. deprecated:: ':
                del child.doc[grandchild_name]

    if hasattr(mod, '__all__'):
        for child_name in mod.__all__:
            child = getattr(mod, child_name)
            if isinstance(child, ModuleType):
                child = pdocify(child, prefix=prefix)
                child.supermodule = pdoc_mod
                child.name = prefix + child_name
                pdoc_mod.doc[prefix + child_name] = child

    return pdoc_mod


if __name__ == '__main__':
    pdoc.cli.args.output_dir = sys.argv[1]
    pdoc.cli.args.force = True
    constriction_mod = pdocify(constriction)
    del constriction_mod.doc['constriction']
    pdoc.cli.recursive_write_files(constriction_mod, '.html')
