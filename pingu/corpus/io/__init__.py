from .default import DefaultLoader


def available_loaders():
    """ Return dictionary with the mapping loader-name, loader-class for all available dataset loaders. """
    return {
        DefaultLoader.type(): DefaultLoader
    }


def create_loader_of_type(type_name):
    """ Return an instance of the loader of the given type. """
    loaders = available_loaders()

    if type_name in loaders.keys():
        return loaders[type_name]()
