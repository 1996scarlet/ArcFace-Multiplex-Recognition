// Modules to control application life and create native browser window
const {app, BrowserWindow} = require('electron')
const path = require('path')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow

function createWindow () {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 730,
    frame: false,
    resizable: false,
    webPreferences: {
      // preload: path.join(__dirname, 'preload.js'),
      preload: 'static/js/preload.js',
      zoomFactor: 0.66
    }
  })

  let contents = mainWindow.webContents

  // and load the index.html of the app.
  mainWindow.loadFile('index.html')

  // Open the DevTools.
  // mainWindow.webContents.openDevTools()

  // Emitted when the window is closed.
  mainWindow.on('closed', function () {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null
  })

  mainWindow.on('enter-full-screen', () => {
    contents.setZoomFactor(1.0)
  })
  
  mainWindow.on('leave-full-screen', () => {
    contents.setZoomFactor(0.66)
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', function () {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (mainWindow === null) createWindow()
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.

const server = require('http').createServer(app);
const io = require('socket.io')(server);

var lunatic_current_ip = '';

io.of('/remilia').on('connection', (socket) => {
  console.log('a lunatic client connected');

  socket.on('change_ip', (msg) => {
      lunatic_current_ip = msg;
      console.log(msg);
  });

  socket.on('frame_data', (msg) => {
      socket.emit('response', lunatic_current_ip);
      if (msg.frame != 0) {
          socket.broadcast.emit('frame_download', {
              'frame': msg.frame.toString('base64'),
              'result': msg.result
          });
      }
  });

  socket.on('result_data', (msg) => {
      socket.broadcast.emit('result_download', {
          'image': msg.image.toString('base64'),
          'time': msg.time,
          'name': msg.name,
          'prob': msg.prob,
          'ip': msg.ip
      });
  });

  socket.on('disconnect', () => { console.log('a lunatic client disconnected') });
});

server.listen(6789, () => console.log('listening on *:6789'));