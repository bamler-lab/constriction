import pdoc
import pdoc.cli
import constriction

from types import ModuleType
import sys

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
    pdoc.cli.recursive_write_files(pdocify(constriction), '.html')
