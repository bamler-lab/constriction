import pdoc
import pdoc.cli
import constriction
from types import ModuleType

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
    pdoc.cli.args.output_dir = 'html'
    pdoc.cli.args.force = True
    pdoc.cli.recursive_write_files(pdocify(constriction), '.html')

# def recursive_htmls(mod):
#     pdoc_mod = pdoc.Module(mod)
#     if hasattr(mod, '__all__'):
#         for submod_name in mod.__all__:
#             submod = getattr(constriction, submod_name)
#             if isinstance(submod, ModuleType):
#                 pdoc_mod.doc[submod_name] = pdoc.Module(submod)
#     yield pdoc_mod.name, pdoc_mod.html()
#     for submod in pdoc_mod.submodules():
#         yield from recursive_htmls(submod)

# for mod in modules:
#     for module_name, html in recursive_htmls(mod):
#         ...  # Process
