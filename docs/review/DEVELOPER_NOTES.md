## Developers personal notes and observations

### security
- bring out system specific variables from code to .env
- bring out referred folders from code to .env

### test files
- get model values from .env as opposed to being declared explicitly

### on android the top bar is crowded. from left to right there is
x remove assistant text and icon
x remove new chat. the button exists in the sidebar
x microphone button still top right: red when active, green when transcribing, gray when idle.
x this means that Computer, transcribing text will be removed, only microphone status.

### microphone status indicator
- currently not working properly.
- colors for following states:
-- grey/no color when idle
-- red when recording
-- blue when actively listening
-- green when transcribing

### voice
x there has got to be an already built function that processes the text that goes to voice to fix/clean issues like:
x spelling out 40,075 properly - e.g. forty thousand seventy five as opposed to forty, zero seven five
x voice output text should be processed to a text that is talkable

### music playback
- review strawberry access as opposed to mpd's own library management
- playlist generation still makes a short list in some case. eg play michael jackson gives one song by him. why is that?
- play Heavy Metal gives a could not reach MPD - is it running error.
- Play 50s seems to be working somewhat
- i have the genres.txt and also the artist names, 'play 50s' should go through genres list first and if the genre doesnt exist it looks for artist

### Tool calls
- when writing i could use / to trigger
- but how would voice trigger a tool call?

### sidebar
- hide/show sidebar: works for phone, also implement for desktop

### chats window
- clicking on new in chat window still spawns a new session each time (still happening)
- review how each chat title should be generated. last chat is no longer feasible.
- delete is the letter x not delete
- future: delete moves to chat settings menu
- future: rename chat, also from chat settings menu
- future: chat settings menu will pop up from a vertical three dot menu on the right of each chat title.
- future: chat settings menu will include delete, rename, and future features like pin to top, archive, etc.

### message window
- stop chat feature

### music window
- display format in artist - song, not song - artist
- artist - song no more than one line
- future: music could be a top or bottom bar?
