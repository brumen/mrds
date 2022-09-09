""" Decorator for storing the fields of the object.

    mongo database is called 'test_store'.
"""

from typing  import Dict, Any
from pymongo import MongoClient


class StoreException(Exception):
    pass


def stored():
    """ decorator for the
    """
    def tags_decorator(func):
        func._tag = 'stored'
        return func
    return tags_decorator


class Store:
    """ Object which handles the storing of the objects, and loading from the mongo db.
    """

    def mongo_db(self) -> str:
        """ Stores the name of the mongo db where the objects are stored.

        """
        raise NotImplementedError('Function mongo_db not implemented')

    def name(self) -> str:
        """ Stores the name of the object.
        """

        return self.__class__

    def _get_all_stored(self) -> Dict[str, Any]:
        """Searches through the class namespace returns all methods with the correct label.
        """

        return {func: getattr(self.__class__, func)(self)
                for func in dir(self.__class__)
                if '_tag' in dir(getattr(self.__class__, func)) and getattr(self.__class__, func)._tag == 'stored'}


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

    def write(self):
        """ Writes the object to the stored database.
        """

        db = self._client().test_store  # TODO: to be further improved later.
        to_insert = self._get_all_stored()
        to_insert |= {'name': str(self.name)}
        db.stored_objects.insert_one(to_insert)

    def load(self):
        """ How to recreate the objects from the database entry. TODO:
        """

        pass
