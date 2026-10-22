import * as net from 'net';

const connection = net.createServer(socket =>{
    let buffer = "";

    socket.on('data', (data) => {
        buffer += data.toString();
        while(buffer.indexOf("\n") != -1){
            let fullLine = buffer.slice(0,buffer.indexOf("\n"));
            buffer = buffer.slice(buffer.indexOf("\n")+1);
            try{
                let formattedData = JSON.parse(fullLine);
                if(formattedData.cmd == "step"){
                    socket.write(JSON.stringify({ok: true, reply: "game stepped forward"})+ "\n");
                }else if(formattedData.cmd=="reset"){
                    socket.write(JSON.stringify({ok: true, reply: "game reset"})+ "\n");
                }
                else{
                    socket.write(JSON.stringify({ok: false, error: "unknown command"})+ "\n");
                }
            }catch (readError){
                socket.write(JSON.stringify({ok:false, error: String(readError)})+ "\n");
            }
                
        }
    });
    socket.on('error', () => {
        console.error("Error in the connection.");
    });
    socket.on('close', () => {
        console.log("Connection closed.\n");
    });
});

connection.listen(5555, '127.0.0.1', () => {
    console.log("Waiting for connection.\n");
});