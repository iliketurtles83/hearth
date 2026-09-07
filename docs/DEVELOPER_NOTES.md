## Developers personal notes and observations

### update beets
- how to update when i have have made changes to music library?
- currently i am going inside the docker container and then calling 'beet update' (or was there a special flag?)

### security
- bring out system specific variables from code to .env
- bring out hardcoded folders from code to .env

### test files
- get model values from .env as opposed to being declared explicitly

### Tool calls
- write json for weather tool call???


### Task: UI top bar removal **DONE**
- Folder: frontend/
- currently top bar only holds hamburger in top left and microphone in top rigght.
- remove top bar
- hamburger resides in left panel with Hearth written next to it. 
- the microphone will remain top right without top bar

### Task: sidebar visual improvements
- Scope folder: frontend/
- no more borders for songs, chats or memories 
- they have the same color as background.
- when hovering over, highlight entire background of the item
- remove the text 'queue' it is implied.
- song row spacing
- song artist and title on one line

### Task: sidebar menu improvements
- no more total scroll bar
- vertical height is total height
- collapse expand sidebar sections

### Task: per chat settings
- Folder: frontend/
- chat settings menu is accessible by vertical three dot menu on the right of each chat title.
- will pop up a small chat settings menu
- chat settings menu will include delete, rename
- move delete to chat settings menu
- create rename chat, also from chat settings menu

### music window
- Folder: frontend/
- artist - song on one line

### settings menu
- appears when clicking on username .
- in settings we will have: theme (dark/light), display reasoning (on/off), manual beet update button, logout button, manual consolidation?
