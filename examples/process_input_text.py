# encoding: utf-8
from horus.core.service import Core

# = ["\xe5\x81\x9a\xe6\x88\x8f\xe4\xb9\x8b\xe8\xaf\xb4"]
#a = [l[0].decode('utf8')]
#print a[0]

#text = "diego's estees-III @sajdh yo yo go!brow. ha!"
#text = 'Driving , driving , driving away to Phil . Tasty dinner tonight with the Society of Mining and Metallurgy Engineers .'
#text = "paris hilton was once the toast of the town"
text = "james bond is the guy"
#text = u"bullshit about airports/coffee/conferences".encode('utf8')
horus = Core()
horus.annotate(text)
#print horus.get_cv_annotation()