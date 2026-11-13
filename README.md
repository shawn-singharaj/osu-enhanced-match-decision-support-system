# osu-enhanced-match-decision-support-system
OSU Men's Soccer backed project for automated timestamping of match footage utilizing AI tools.

Shawn's Progress:
Current Goal: Draw the goal lines
Limitations: Not always in frame, need to figure out how to "estimate it", eventually create our own grid
Definition of goal line: Angle of 15-75 degrees or 105-165 degrees for goalines, and to not confuse with penalty box, must be touching the top/bottom sidelines. 

TODO:
- Separate function to filter those goal line angles (like it does for the sidelines)
- y = m*y + b to store line
- Identify goaline vs penalty box line
- left or right distinguish
- draw
- how to deal with if not in frame

**How field_mapping.py works:**
**Every frame goes through this pipeline:**
for frame x:
-> filter_grass, green mask "find the pitch, filter by greens" -> white_on_grass, white mask "find the painted sideline borders of the pitch, filter out smaller objects like players wearing white" -> HoughLinesP, segments "which painted pixels form straight lines" -> filtering loop, "use whites found + straight lines to find which straight lines = sidelines" -> smooth, far_line/near_line "steady it across frames" -> draw_line -> out.write = output video

filter_grass - Returns solid white silhouette of pitch:
Takes HSV, called by find_sidelines, which converts the frame from BGR (blue green red). With this, we can more easily mask with the color green to filter only the pitch, which is the largest green object on video.
  Hue: which color from 0-179 in OpenCV, where green = 35-85, OSU home grass = 33
  Saturation: How colorful, 0 = white
  Value: How bright

white_on_grass - Returns a mostly black image with the painted lines in white, plus some noise from jerseys that are only on the pitch:
Takes BGR frame, HSV frame, field from filter_grass. 
The lines will always be brighter than the grass next to it. 

find_sidelines - Returns far and near long sidelines
Hough transform call returns segments with endpoints, voted upon  by every white pixel. Returns an array of (x1, y1, x2, y2) rows.
Take said segments, filter by horizontal lines, create line equation y= mx + b, see longest + closest to edge of field mask, and choose the top and bottom.

Next:
Take segments, filter by vertical lines, create line equation x = my + b, see if line touches both sidelines, return that, handle if one goaline too




