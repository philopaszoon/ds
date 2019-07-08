Quora question pair study - adding some features

quora_count_features_2:
<br>https://github.com/philopaszoon/ds/blob/master/quora_question_pairs/quora_count_features_2.ipynb
* counting words and characters
* exploring the odds of a match by char and word counts


quora_quoted_strings_4:
<br>https://github.com/philopaszoon/ds/blob/master/quora_question_pairs/quora_quoted_strings_4.ipynb
* identifying sentences in which there is a quoted string


quora_first_words_5:
<br>https://github.com/philopaszoon/ds/blob/master/quora_question_pairs/quora_first_words_5.ipynb
* In many languages, the first word of a question is often a marker of what kind of question it is.  For example the question, "Who was your first grade teacher?", is obviously a question about a person.  It's also an example of a placeholder question, where placeholders like Who, Where, Why, What, When take the grammatical place of the object of the verb in the sentence.  This can be better seen by rewriting the question in a declarative form, "Your first grade teacher [was] Who".  Since a question about a person is unlikely to match a question about a place, simply recording what the first words are can be useful.

* First words can also be helpful in distinguishing yes/no questions from placeholder questions.  Typically, questions that begin with a "to be" verb are yes/no questions.  ex: "Are we there yet?".  This class also includes words like Do, Would, Have, and others.
<pre>
        Inspector Clouseau      -> "[Does] your dog bite?"
        Man behind the counter  -> "No"
        <i>[ Peter Sellers reaches down to pet the dog, which immediately bites him]</i>
        Inspector Clouseau      -> "I thought you said your dog did not bite"
        Man behind the counter  -> "That is not my dog"
        https://www.youtube.com/watch?v=ui442IDw16o
</pre>
* Of course, the formatting of real questions is much more diverse than in these toy examples.  The word "who" might be replaced by "what driver" or "which president", and the words that indicate what the question is might appear later in the sentence.  There are a number of ways that this section can be improved, though at the cost of greater code complexity as one finds oneself unraveling prepositional phrases, compound nouns and other grammatical features.

* You may notice that there's a lot more than first words being recorded in this notebook.  There are not only parts of speech related to first words, but also a group of columns to describe the “head” of the first word.  A “head” is a root dependency of a given word.  For example, in the sentence, “Who is the fastest runner on earth?”, the question word (Who) has as its head word the main verb (is), which also happens to be the root of the whole sentence.  

word    | orth    | lemma  | tag | dep  | pos  | head   | head_pos | tree
--------|---------|--------|-----|------|------|--------|----------|--------------------------------
Who     | Who     | who    | WP  | attr | NOUN | is     | VERB     | [Who]
is      | is      | be     | VBZ | ROOT | VERB | is     | VERB     | [Who, is, the, fastest, runner, on, earth, ?]
the     | the     | the    | DT  | det  | DET  | runner | NOUN     | [the]
fastest | fastest | fast   | JJS | amod | ADJ  | runner | NOUN     | [fastest]
runner  | runner  | runner | NN  | attr | NOUN | is     | VERB     | [the, fastest, runner, on, earth]
on      | on      | on     | IN  | prep | ADP  | runner | NOUN     | [on, earth]
earth   | earth   | earth  | NN  | pobj | NOUN | on     | ADP      | [earth]

Not surprisingly, there’s a lot of information in the relationships among words in a sentence.  One-hot encoding is a bit of a bottleneck in this regard, making it difficult to represent the full range of these relationships in compact form.  I focused on capturing the dependency of the ‘head’ (mainly to know if it’s the sentence ROOT or not) and on recording whether the head is a “to be” verb or not.  Doing more than that would have exceeded the limitations of my hardware.  

I want to note, however, that providing the information to the random forest in categorical form (as it appears in cell # 8 in the notebook rather than one-hot encoded) produces nearly the same results as one-hot encoding.  It’s a flawed representation from the point of view of what the algorithm is doing, but it’s also more complete.  It seems possible that those two facts balance each other out. 


