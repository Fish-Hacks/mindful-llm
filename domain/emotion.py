from dataclasses import dataclass
from json import loads
from typing import Optional, List


@dataclass
class Emotion:
    '''
    Represents an emotion entry with its date and an optional description.

    Attributes:
        date (str): The date when the emotion was recorded.
        emotion (str, optional): The type of emotion. Defaults to "No Emotion".
        description (Optional[str], optional): A description or reason for the emotion. 
                                               Defaults to "No Description".

    Methods:
        from_json: Parses a JSON string and returns a list of Emotion instances.
    '''

    date: str
    emotion: str = "No Emotion"
    description: Optional[str] = "No Description"

    @staticmethod
    def from_json(json: str) -> List['Emotion']:
        '''
        Parses a JSON string to create a list of Emotion instances.

        Args:
            json (str): A JSON string representation of a list of emotions. Each emotion should have attributes 
                        that match the Emotion class.

        Returns:
            List[Emotion]: A list of Emotion instances.
        '''
        data = loads(json)

        return [Emotion(**entry) for entry in data]
