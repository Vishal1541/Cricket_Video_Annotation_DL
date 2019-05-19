import urllib, json

for page in range(5):

    # AUS Innings
    url = "http://site.web.api.espn.com/apis/site/v2/sports/cricket/19059/playbyplay?contentorigin=espn&event=1168247&page=%s&period=2&section=cricinfo" % (page + 1)

    # INDIA Innings
    # url = "http://site.web.api.espn.com/apis/site/v2/sports/cricket/19059/playbyplay?contentorigin=espn&event=1168247&page=%s&period=1&section=cricinfo" % (page + 1)

    response = urllib.urlopen(url)

    data = json.loads(response.read().decode())

    L = len(data["commentary"]["items"])

    for l in (data["commentary"]["items"]):
        data = l["shortText"].split(", ")[1] + " " + l["text"] + "."
        print data