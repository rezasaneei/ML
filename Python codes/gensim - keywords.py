
"""
Created on Mon Jan 21 21:25:01 2019

@author: rezas
"""

from gensim.summarization import keywords

docs = ['At the close of WWII, a young nurse tends to a badly-burned plane crash victim. His past is shown in flashbacks, revealing an involvement in a fateful love affair.'
,'The sweeping expanses of the Sahara are the setting for a passionate love affair in this adaptation of Michael Ondaatjes novel. A badly burned man, Laszlo de Almasy, is tended to by a nurse, Hana, in an Italian monastery near the end of World War II. His past is revealed through flashbacks involving a married Englishwoman and his work mapping the African landscape. Hana learns to heal her own scars as she helps the dying man.'
,'Beginning in the 1930s, The English Patient tells the story of Count Almásy who is a Hungarian map maker employed by the Royal Geographical Society to chart the vast expanses of the Sahara Desert along with several other prominent explorers. As World War II unfolds, Almásy enters into a world of love, betrayal, and politics that is later revealed in a series of flashbacks while Almásy is on his death bed after being horribly burned in a plane crash.'
,'A burn victim, a nurse, a thief, and a sapper find themselves in each others company in an old Italian villa close to the end of World War II. Through flashbacks, we see the life of the burn victim, whose passionate love of a woman and choices he made for her ultimately change the lives of one other person in the villa. Not only is this film a search for the identity of the English patient, but a search for the identities of all the people in the quiet old villa.'
,'In a crumbling villa in WWII Italy, during the final days of the European campaign, a young, shell-shocked war nurse (Hana) remains behind to tend her doomed patient - a horribly burned pilot. Through the gradual unraveling of his life and the appearance of an old family friend (Caravaggio) and a young Sikh sapper (Kip), the question of identity is explored.'
]

print(keywords(docs[0]))
print(keywords(docs[1]))
print(keywords(docs[2]))
print(keywords(docs[3]))
print(keywords(docs[4]))
