from onvif import ONVIFCamera

ip = "192.168.1.87"
port = 5000
user = "admin"
password = "Haithy123"

cam = ONVIFCamera(ip, port, user, password)

media = cam.create_media_service()
profiles = media.GetProfiles()

for p in profiles:
    stream = media.GetStreamUri({
        'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
        'ProfileToken': p.token
    })
    print(stream.Uri)