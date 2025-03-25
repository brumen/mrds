""" Decorator for storing the fields of the object.

    mongo database is called 'test_store'.
"""

import logging

from functools import partial, wraps
from typing    import Dict, Any
from pymongo   import MongoClient

_logger = logging.getLogger(__name__)


class StoreException(Exception):
    pass


def stored():
    """ decorator for the
    """

    def tags_decorator(func):

        func._tag = 'stored'  # tag to identify which value is stored.
        return func

    return tags_decorator


class Store:
    """ Object which handles the storing of the objects, and loading from the mongo db.
    """

    @classmethod
    def stored_class(cls):

        def wrapper(fn):
            if hasattr(cls, '_decorated_fns'):
                cls._decorated_fns.append(fn)
            else:
                cls._decorated_fns = [fn]

            return fn

        return wrapper

    def mongo_db(self) -> str:
        """ Stores the name of the mongo db where the objects are stored.

        """
        raise NotImplementedError('Function mongo_db not implemented')

    def name(self) -> str:
        """ Stores the name of the object.
        """

        return self.__class__

    def _is_stored(self, func_name : str) -> bool:
        """ Is func_name decorated as stored.

        :param func_name: name of the method
        :returns: indicator whether the func_name is stored.
        """

        return '_tag' in dir(getattr(self.__class__, func_name)) and getattr(self.__class__, func_name)._tag == 'stored'


    def _get_all_stored(self) -> Dict[str, Any]:
        """Searches through the class namespace returns all methods with the correct label.
        """

        stored_objs = {}
        for func in dir(self.__class__):
            if self._is_stored(func):
                class_fct = getattr(self.__class__, func)
                if isinstance(class_fct, property):  # property is handled
                    stored_objs[func] = getattr(self, func)
                else:
                    stored_objs[func] = getattr(self, func)()  # evaluate the method.

        return stored_objs

    def _get_all_stored_new(self) -> Dict[str, Any]:
        """Searches through the class namespace returns all methods with the correct label.
        """

        stored_objs = {}
        for func in self._decorated_fns:  # only properties can get decorated
            stored_objs[str(func)] = func.fget(self)

        return stored_objs


    # TODO: REWORK the __client caching.
    __client = None

    @classmethod
    def _client(cls):
        """ Storing the client, so there is only one per class.
        """

        if cls.__client:
            return cls.__client

        cls.__client = MongoClient()
        return cls.__client

    def write_old(self, path : str = '/Objects/object1') -> None:
        """ Writes the object to the stored database.

        :path: path of the object to write.
        :returns: writes the object to database.
        """

        db = self._client().test_store  # TODO: to be further improved later.
        to_insert = self._get_all_stored()
        to_insert |= {'name': str(self.name)
                      , 'path': path
                      , }
        db.stored_objects.insert_one(to_insert)

    def write_new(self, path : str = '/Objects/object1') -> None:
        """ Writes the object to the stored database.

        :path: path of the object to write.
        :returns: writes the object to database.
        """

        db = self._client().test_store  # TODO: to be further improved later.
        to_insert = self._get_all_stored_new()
        to_insert |= {'name': str(self.name)
                      , 'path': path
                      , }
        db.stored_objects.insert_one(to_insert)


    @staticmethod
    def constant_fct(x):
        return x

    # methods that are not relevant for loading.
    _methods_not_relevant = ('_id', 'path', 'name', )

    @classmethod
    def load_old(cls, path : str, *args, **kwargs):
        """ Recreate the object from the database, loading from path.

        :path: path of the object to search for.
        """

        db = cls._client().test_store  # TODO: to be further improved later.
        object_dict = list(db.stored_objects.find({'path': path}))

        reconstructed_obj = cls(*args, **kwargs)  # initial constructor

        if not object_dict:  # if nothing is found.
            return reconstructed_obj

        if len(object_dict) > 1:
            _logger.warn(f'More than one object found w/ path {path}. Using the first one')

        # recreate everything except for _id, path, name
        for method_name, method_value in object_dict[0].items():  # TODO: NOT ONLY THE FIRST ONE
            if method_name not in cls._methods_not_relevant:
                if not isinstance(getattr(cls, method_name), property):  # not a property
                    setattr(reconstructed_obj, method_name, partial(cls.constant_fct, method_value))  # IMPORTANT: partial has to be here.
                else:  # is a property, have to do differently.
                    setattr(reconstructed_obj, method_name, method_value)

        return reconstructed_obj

    @classmethod
    def load_new(cls, path : str, *args, **kwargs):
        """ Recreate the object from the database, loading from path.

        :path: path of the object to search for.
        """

        db = cls._client().test_store  # TODO: to be further improved later.
        object_dict = list(db.stored_objects.find({'path': path}))

        reconstructed_obj = cls(*args, **kwargs)  # initial constructor

        if not object_dict:  # if nothing is found.
            return reconstructed_obj

        if len(object_dict) > 1:
            _logger.warn(f'More than one object found w/ path {path}. Using the first one')

        # recreate everything except for _id, path, name
        for method_name, method_value in object_dict[0].items():  # TODO: NOT ONLY THE FIRST
            if method_name not in cls._methods_not_relevant:
                # TODO: ONLY STORING PROPERTIES
                setattr(reconstructed_obj, method_name, method_value)

        return reconstructed_obj
