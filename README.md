# osu-enhanced-match-decision-support-system
OSU Men's Soccer backed project for automated timestamping of match footage utilizing AI tools.

Shawn's Progress:
Goal: Draw the goal lines
Limitations: Not always in frame, need to figure out how to "estimate it", eventually create our own grid
Definition of goal line: Angle of 15-75 degrees or 105-165 degrees for goalines, and to not
confuse with penalty box, must be touching the top/bottom sidelines. 

TODO:
- Separate function to filter those goal line angles (like it does for the sidelines)
- y = m*y + b to store line
- Identify goaline vs penalty box line
- left or right distinguish
- draw
- how to deal with if not in frame
