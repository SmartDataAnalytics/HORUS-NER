The token metadata class. An important side note: the high level of noisiness in tweet data makes it difficult to correctly map tokens, labels and data coming from external tokenizers (e.g., when obtaining POS taggers for a ConLL file without such information). This alignment is already tricky due to the nature of the data, but becomes even more complex as in some cases, we may have a token with contains 2 levels of HTML encoded information.

#### PROBLEM 1

For instance,

```s1 = 'Going on a &amp;#39 ;d ate&amp;#39; with Whitney tonight, :P. Haha.' -> Original from (Ritter, 2011)```

-- 1 time ```html.escape()``` produces

```s2 = 'Going on a &#39 ;d ate&#39; with Whitney tonight, :P. Haha.'```

-- +1 time ```html.escape()``` produces

```s3 = 'Going on a \' ;d ate\' with Whitney tonight, :P. Haha.'```

In the above original sentence (```s1```) the token is represented as = ```&amp;#39 0``` in the respective CoNLL file.
However, we see here 2 levels of encoding:

```
E1 = token                               = '&amp;#39' -> len() = 8
E2 = html.unescape(token)                = '&#39'     -> len() = 4
E3 = html.unescape(html.unescape(token)) = '\''       -> len() = 2
```

For a token alignment based on char-level index (e.g., ```token```, ```begin_index```, ```end_index```), this implies 
in having the wrong (begin, end) indexes, as the ```len(token)``` may vary (see above).

However, we need to keep indexes consistent for token/labels alignment but depending on the tokenizer's
implementation one uses, the outcome may be different from what we expect.
A real example for s1 is as follows (tokenizer = TweetNLP, (Gimpel et al. 2011)):

```
[('Going', 'VBG'), ('on', 'IN'), ('a', 'DT'), ('&', 'CC'), ('#39', 'HT'), (';d', 'UH'), ('ate', 'UH'),
 ("'", "''"), ('with', 'IN'), ('Whitney', 'NNP'), ('tonight', 'NN'), (',', ','), (':P', 'UH'), ('.', '.'),
 ('Haha', 'UH'), ('.', '.')]
```

 It can be seen that the tokenizer messed up the original sentence, as so the token indexes (begin, end).
 To avoid such cases and guarantee future tokenizers interoperability/non-dependency, we perform a short-loop
 to make sure we have always the correct unescaped token, so that indexes are always consistent (see E3).

#### PROBLEM 2

 There is still a specific case where we can not do much, due to tokenization mistaken in the original CoNLL
 file, where there are 2 tokens to represent a single unit, when decoded. For instance, in the sentence:

```
 s = 'Ah,thathasseriouslymademyday.:D@chrisbrown,iloveyousomuch&hearts;'
```

 The last 2 tokens in the CoNLL file are:
 ```
 ...
 T_(n-1) = 'iloveyousomuch&amp;hearts'	O
 T_n = ';'	'O'
```

 This is wrong by definition, as the token would only be decoded correctly if a single unit, i.e.,

```
 html.unescape('&amp;') => '&'
 html.unescape('&hearts;') => '♥'
```
 The final correct sentence should then be: ```'iloveyousomuch♥'```

```
 O1 = html.unescape('iloveyousomuch&amp;hearts;') = 'iloveyousomuch&hearts;'
 O2 = html.unescape(O1) = 'iloveyousomuch♥'
```

Therefore, the error is in the construction of the CoNLL file, which should not be split into these 2 tokens
(```'iloveyousomuch&amp;hearts'```) and (```';'```) as they are - technically - a single token.

To avoid this, we must keep punctuation with spaces in the sentence. For instance:

```'@1STLADYNHEELS Alright , I'll get there around that time . I'll call/text you when I arrive .'```

instead of

```'@1STLADYNHEELS Alright, I'll get there around that time. I'll call/text you when I arrive.'```