import time

orig_time = int(round(time.time() * 1000))
salarypermilisec= 365000/31/24/60/60/1000

try:
    while True:
        time.sleep(0.01)
        new_time = int(round(time.time() * 1000))
        difference = (new_time - orig_time) * salarypermilisec
        print("\r %.4f HUF" % difference, end="")
except KeyboardInterrupt:
    print("\n STOPPED")