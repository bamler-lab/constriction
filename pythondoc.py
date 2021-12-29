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
    if hasattr(mod, '__all__'):
        for submod_name in mod.__all__:
            submod = getattr(mod, submod_name)
            if isinstance(submod, ModuleType):
                child = pdocify(submod, prefix=prefix)
                child.supermodule = pdoc_mod
                child.name = prefix + submod_name
                pdoc_mod.doc[prefix + submod_name] = child

    return pdoc_mod


if __name__ == '__main__':
    pdoc.cli.args.output_dir = sys.argv[1]
    pdoc.cli.args.force = True
    constriction_mod = pdocify(constriction)
    del constriction_mod.doc['constriction']
    pdoc.cli.recursive_write_files(constriction_mod, '.html')
