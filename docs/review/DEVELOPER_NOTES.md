## developers personal notes and observations

### security
- bring out system specific variables from code to .env
- bring out referred folders from code to .env

### test files
- get model values from config as opposed to being declared explicitly


### android issue

#### issue 2: on android the top bar is crowded. from left to right there is
- menu hamburger
- icon and assistant text
- voice chat indicator
- new chat button
- microphone button

#### issue 2 resolution:
- remove assistant text and icon
- microphone top right is ok for now
- remove new chat. the button exists in the sidebar
- voice chat indicator as icon, possibly pulsing microphone icon when it speaks.


### voice
there has got to be an already built function that processes the text that goes to voice to fix/clean issues like:
- spelling out 40,075 properly - e.g. forty thousand seventy five as opposed to forty, zero seven five
- not spell out asterisks if they exist in the text.

### music playback
- playlist generation still makes a short list in some case. eg play michael jackson gives one song by him
- need to figure out what is a good way here? include artist genre keywords to trigger artist genre? or smtn else
- i have the genre-tree.txt and also the artist names, 'play 50s' should go through genres list first and if the genre doesnt exist it looks for artist

### sidebar
- hide/show sidebar: works for phone, also implement for desktop

#### chats window
- clicking on new in chat window spawns a new session each time (still happening)
- review how each chat title should be generated. last chat is no longer feasible.
- delete is the letter x not delete
- future: delete moves to chat settings menu
- future: rename chat, also from chat settings menu
- chat settings menu will be a vertical three dot menu on the right of each chat title.
- chat settings menu will include delete, rename, and future features like pin to top, archive, etc.

#### message window
- stop chat feature

#### music window
- display format in artist - song, not song - artist
- artist - song no more than one line
- future: music could be a top or bottom bar?
