from datamuse import Datamuse
api = Datamuse()
print(api.words(rel_rhy='sign', max=5))  # words that rhyme with "ninth"

print(api.words(rel_rhy='orange', max=5))  # words that rhyme with "orange"

print(api.words(rel_jja='yellow', max=5))  # things often described as "yellow"

print(api.suggest(s='foo', max_results=10))  # completion suggestions for "foo"
