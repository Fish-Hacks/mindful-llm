from dataclasses import dataclass
from typing import Optional


@dataclass
class Emotion:
    '''
    Represents an emotion entry with its date and an optional description.

    Attributes:
        date (str): The date when the emotion was recorded.
        emotion (str, optional): The type of emotion. Defaults to "No Emotion".
        description (Optional[str], optional): A description or reason for the emotion. 
                                               Defaults to "No Description".
    '''
    date: str
    emotion: str = "No Emotion"
    description: Optional[str] = "No Description"
