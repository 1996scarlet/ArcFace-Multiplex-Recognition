// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.
var socket = io("http://127.0.0.1:6789/remilia");
socket.on('frame_download', (msg) => frame_queue.push(msg));
socket.on('result_download', (result) => drawResult(result));

function drawResult(result) {
    // socket.emit('change_ip', result.ip);
    let lunatic_result = lunatic_judge(result);
    let part_list = document.getElementById(lunatic_result[0]), txt_li = lunatic_result[1];

    if (part_list.children.length > 3) part_list.children[0].remove();
    part_list.innerHTML += txt_li;
};

window.onload = function () {
    var frame_ctx = document.getElementById("c").getContext("2d");
    frame_ctx.font = "40px Black";
    frame_ctx.lineWidth = "4";
    frame_ctx.strokeStyle = "Yellow";

    [0, 1, 2, 3, 4].forEach((value, index) =>
        document.getElementById('camera_list_top').innerHTML += "<button onclick='sendIP(this)' name='Camera_Button' id='bt1' class='h-100 w-25 btn btn-outline-info' value='" + value + "'>Camera " + value + "</button>");

    [5, 6, 7, 8, 9].forEach((value, index) =>
        document.getElementById('camera_list_down').innerHTML += "<button onclick='sendIP(this)' name='Camera_Button' id='bt1' class='h-100 w-25 btn btn-outline-info' value='" + value + "'>Camera " + value + "</button>");

    setInterval(function () {
        let msg = frame_queue.shift();
        if (msg) {
            image.onload = drawFrame(msg, frame_ctx);
            image.src = getUrl(msg.frame);
        };
    }, 40);
}

var websocketClose = () => socket.close();
var threshold = 0.97, img_width = 1121, img_height = 672, input_size = 672;
var frame_queue = [], prefix = 'data:image/jpeg;base64,';
var x_scale = img_width / input_size, y_scale = img_height / input_size;
var image = new Image(), g_res = [], counter = 0, danger_counter = 0;
var getUrl = (base_string) => { return prefix + base_string };
var sendIP = (m_btn) => socket.emit('change_ip', m_btn.value);


var drawRect = (result, context) => result.forEach(poi => {
    var xmin = poi[0] * x_scale, ymin = poi[1] * y_scale;
    context.strokeText((poi[4] * 100).toFixed(2), xmin, ymin);
    context.strokeRect(xmin, ymin, poi[2] * x_scale - xmin, poi[3] * y_scale - ymin);
});

var drawFrame = (msg, context) => {
    context.drawImage(image, 0, 0, img_width, img_height);
    getGlobalResult(msg.result);
    drawRect(g_res, context);
}

var getGlobalResult = (result) => {
    if (result) {
        g_res = result;
        counter = 0;
    } else if (++counter > 10) {
        g_res = [];
        counter = 0;
    }
};

// -----------------------LUNATIC----------------------
var lunatic_judge = (result) => {
    let pl, tl;
    if (result.prob > threshold) {
        pl = "worker_list";
        tl = "<li class='media mx-4 py-2 text-white'>\
            <img class='align-self-center mr-3'  width='60px' height='60px' src=" + getUrl(result.image) +
            " /><div class='media-body'><h5>" + result.name + " -- Credibility: " + (result.prob * 100).toFixed(2) + "%</h5><span>DateTime: " + result.time + "</span></div></li>";
    } else {
        pl = "suspect_list";

        tl = "<li class='media mx-4 py-2 text-white'>\
        <img class='align-self-center mr-3'  width='60px' height='60px' src=" + getUrl(result.image) +
        " /><div class='media-body'><h5>Suspicious: " + (100 - result.prob * 100).toFixed(2) + "%</h5><span>DateTime: " + result.time + "</span></div></li>";
    }
    return [pl, tl]
}
