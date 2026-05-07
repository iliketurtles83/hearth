## Developers personal notes and observations

### security
- bring out system specific variables from code to .env
- bring out hardcoded folders from code to .env

### test files
- get model values from .env as opposed to being declared explicitly

### music playback
- review strawberry access versus mpd's own library management (if there is one)
- consider importing to mdb (if mdb has a library management system)
- playlist generation still makes a short list in some case. eg play michael jackson gives one song by him. why is that?
- play Heavy Metal gives a could not reach MPD - is it running error.
- Play 50s seems to be working somewhat
- i have the genres.txt and also the artist names, 'play 50s' should go through genres list first and if the genre doesnt exist it looks for artist
- should genres.txt be moved?
- how does MPD work? What can it handle? 
- i see music and playlists in mpd folder. how does that work?
- write json for music tool call
- does tool call replace existing regex?

### Tool calls
- write json for weather tool call

## UI

### microphone status indicator
x currently not working properly. just goes to red when clicked and stays there.
x should be colors for following states:
x grey/no color when idle
x red when recording
x blue when actively listening
x green when transcribing
x clicking on it toggles between idle and recording
- add one for voice generation as well, maybe yellow? or orange? 


### sidebar
x hide/show sidebar: works for phone, also implement for desktop

### chats window
x clicking on new in chat window still spawns a new session each time (still happening)
x review how each chat title should be generated. last chat is no longer feasible.
x delete is the letter x not delete
- future: delete moves to chat settings menu
- future: rename chat, also from chat settings menu
- future: chat settings menu will pop up from a vertical three dot menu on the right of each chat title.
- future: chat settings menu will include delete, rename, and future features like pin to top, archive, etc.

### message window
x stop chat feature

### music window
x display format in artist - song, not song - artist
- artist - song on one line
- future: music could be a top or bottom bar? whats the industry standard in this case? perhaps an expandable that shows queue when clicked?
