# Sachin Tendulkar — 8 AV explanations + Haiku claims + Δ
Activation at the last token. Δ = mse(claim's sentence removed) − mse(intact). **Δ<0 = removal IMPROVED reconstruction.**
## The passage
> When Australia toured India in 2001, in the first test at Mumbai, Tendulkar scored 76 and 65, with no one else from India crossing 50. India ended up losing the test. On the final day of the 2nd test, the famous Kolkata Test against Australia in 2001, Tendulkar took three wickets, including the key wickets of Matthew Hayden and Adam Gilchrist, who were centurions in the previous Test. His three wickets haul helped India win the match. In the 3rd test at Chennai, Tendulkar top scored for India with 126 helping the team to get to a total of 501. India would go on to win the test match narrowly by just 2 wickets and clinch the test series 2–1. This series is regarded as one of the greatest series of the century.

---
## k=0   (intact mse 0.00880)
### AV wrote
> Sports quiz/trivia format with informational answers, establishing context of a cricket/football player's career achievements in Indian cricket.
> The sentence "This World Cup run was remarkable as he was the best batsman...This decade was the golden era" signals a concluding summary statement wrapping up the quiz description.
> Final sentence "This was a memorable period...this tournament was a landmark achievement" ends a closing remark, strongly expecting a closing statement like "
> What do you think?" or "During this period..." or " Tendulkar was also involved." or "Was this the role of Sachin Tendulkar?" or "So, what statistics are relevant?" or "In this match."

### Claims  (Δ)
- `+0.00053` `THEME/format` The text is in sports quiz/trivia format.
- `+0.00053` `THEME/topic` The text discusses a cricket/football player's career achievements in Indian cricket.
- `+0.00053` `THEME/structure` The text provides informational answers within its quiz format.
- `+0.00149` `ENTITY/person` The text mentions Sachin Tendulkar.
- `+0.00020` `DETAIL/quote` The text contains the sentence "This World Cup run was remarkable as he was the best batsman...This decade was the golden era".
- `+0.00135` `DETAIL/quote` The text contains the sentence "This was a memorable period...this tournament was a landmark achievement".
- `+0.00020` `DETAIL/quote` The text contains the phrase "World Cup run".
- `+0.00020` `DETAIL/quote` The text contains the phrase "golden era".
- `+0.00135` `DETAIL/quote` The text contains the phrase "landmark achievement".
- `+0.00135` `DETAIL/quote` The text contains the phrase "memorable period".
- `+0.00020` `DETAIL/quote` The text references the phrase "best batsman".
- `+0.00020` `ENTITY/event` The text discusses a World Cup.
- `+0.00135` `ENTITY/event` The text discusses a tournament.
- `+0.00053` `DETAIL/statistic` The text contains discussion of statistics relevant to the player's career.
- `+0.00149` `ENTITY/event` The text mentions a match.
- `+0.00053` `DETAIL/quote` The text establishes the context of establishing context of a cricket/football player's career achievements.
- `+0.00020` `THEME/era` The text references a decade as a notable period.
- `+0.00020` `DETAIL/quote` The text contains a concluding summary statement wrapping up the quiz description.
- `+0.00135` `DETAIL/quote` The text contains a closing remark at the end.
- `+0.00149` `DETAIL/quote` The text implies expectations for a closing statement such as "What do you think?"
- `+0.00135` `DETAIL/quote` The text implies expectations for a closing statement such as "During this period..."
- `+0.00149` `DETAIL/quote` The text implies expectations for a closing statement such as "Was this the role of Sachin Tendulkar?"
- `+0.00149` `DETAIL/quote` The text implies expectations for a closing statement such as "So, what statistics are relevant?"
- `+0.00020` `DETAIL/quote` The text implies expectations for a closing statement such as "In this match."
- `(unmatched)` `DETAIL/quote` The text suggests possible involvement of Tendulkar in a period being discussed.

---
## k=1   (intact mse 0.00865)
### AV wrote
> Informational/biographical article structure: factual cricket answer format concludes with a summary statement about Sachin Tendulkar's achievements.
> The phrase "This series was the peak of his career. This tournament saw him…." signals a concluding remark, suggesting a trivia or quiz question framing.
> Final sentence "This series was the peak of his career. This tournament was historic" ends a factual closing statement, implying either a question prompt or elaboration — likely "What role did he play?" or "During this period..." or "
> Sachin Tendulkar..." or "Is this correct?" or "Besides this, what other contribution..." serving as a trivia question about the Indian cricket team.

### Claims  (Δ)
- `+0.00065` `THEME/format` The text is an informational/biographical article structure
- `+0.00065` `THEME/format` The text uses a factual cricket answer format
- `+0.00065` `THEME/structure` The text concludes with a summary statement about Sachin Tendulkar's achievements
- `-0.00011` `DETAIL/quote` The text contains the phrase 'This series was the peak of his career. This tournament saw him....'
- `-0.00011` `THEME/structure` The text contains a concluding remark signaled by the phrase about the series being the peak of his career
- `+0.00068` `THEME/format` The text suggests a trivia or quiz question framing
- `+0.00083` `DETAIL/quote` The text contains the final sentence 'This series was the peak of his career. This tournament was historic'
- `+0.00174` `THEME/structure` The text ends with a factual closing statement
- `+0.00065` `ENTITY/person` The text discusses Sachin Tendulkar
- `+0.00027` `ENTITY/team` The text is about the Indian cricket team

---
## k=2   (intact mse 0.00841)
### AV wrote
> Informational/sports biography format: answer answers factual questions about cricket legend Sachin Tendulkar in a descriptive tone.
> The sentence "This period was a golden period for Indian cricket. This series saw him complete 9000 runs" signals a concluding trivia/summary statement about the above list.
> Final sentence "This was one of the greatest moments in his career" ends a closing remark — strongly expects a question or statement about the achievement, likely "What role did he play?" or "During this series?" or " Tendulkar was..." or "
> Was this correct?" or "Which of these statistics are you referencing?" or "In what context did he..."

### Claims  (Δ)
- `+0.00038` `THEME/format` The text is in an informational/sports biography format
- `+0.00038` `THEME/topic` The text answers factual questions about cricket legend Sachin Tendulkar
- `+0.00038` `THEME/register` The text uses a descriptive tone
- `+0.00038` `ENTITY/person` The text mentions Sachin Tendulkar
- `+0.00066` `DETAIL/quote` The text contains the sentence "This period was a golden period for Indian cricket."
- `+0.00052` `DETAIL/quote` The text contains the sentence "This series saw him complete 9000 runs"
- `+0.00052` `DETAIL/number` The text mentions 9000 runs
- `+0.00288` `DETAIL/quote` The text contains the sentence "This was one of the greatest moments in his career"
- `+0.00052` `THEME/structure` The text contains a concluding trivia/summary statement

---
## k=3   (intact mse 0.00917)
### AV wrote
> Informational/biographical tone with factual sports Q&A format, establishing a cricket quiz answer about Virat Kohli's Indian Premier League dominance.
> The sentence "This series was the climax of his career from 2002 to 1999" signals a concluding summary statement, likely referencing the statistics or accolades mentioned.
> Final sentence "This series was the climax of his career" ends a factual closing remark — immediately expects elaboration like "What was his role?" or "Did he achieve this?" or "
> In this context?" or "During this match..." or "Therefore, what do you think of these achievements?" or "Was Kohli also involved."

### Claims  (Δ)
- `+0.00041` `THEME/format` The text has an informational/biographical tone with factual sports Q&A format
- `+0.00041` `ENTITY/person` The text is about Virat Kohli
- `+0.00041` `ENTITY/event` The text discusses Virat Kohli's Indian Premier League dominance
- `+0.00041` `THEME/topic` The text is a cricket quiz answer
- `+0.00006` `DETAIL/quote` The text contains the sentence 'This series was the climax of his career from 2002 to 1999'
- `+0.00006` `THEME/structure` The text contains a concluding summary statement
- `+0.00006` `DETAIL/quote` The text contains the sentence 'This series was the climax of his career'
- `+0.00039` `THEME/structure` The text ends with a factual closing remark

---
## k=4   (intact mse 0.00895)
### AV wrote
> Informational/biographical structure: the answer format signals a factual cricket post, wrapping up a discussion about a legendary Indian batsman.
> The sentence "This series saw him play exceptionally well... This period was crucial to his career." sets up a closing declarative statement about the Indian team's achievements or the statistic.
> Final sentence "This series was a memorable series. It was known as the golden era of cricket" ends a summary statement, immediately expecting a concluding remark like "What role did Virat play?" or "
> During this match" or "Was this correct?" or " Tendulkar was praised" or "In which of these achievements..." or "His role in this?"

### Claims  (Δ)
- `-0.00077` `THEME/format` The text is a factual cricket post **← removal IMPROVED**
- `-0.00077` `THEME/structure` The text wraps up a discussion about a legendary Indian batsman **← removal IMPROVED**
- `-0.00066` `DETAIL/quote` The text contains the sentence "This series saw him play exceptionally well... This period was crucial to his career." **← removal IMPROVED**
- `-0.00077` `ENTITY/person` The text discusses an Indian batsman **← removal IMPROVED**
- `+0.00040` `THEME/topic` The text discusses the Indian team's achievements or a statistic
- `+0.00067` `DETAIL/quote` The text contains the sentence "This series was a memorable series. It was known as the golden era of cricket"
- `+0.00067` `THEME/era` The text references the golden era of cricket

---
## k=5   (intact mse 0.00893)
### AV wrote
> Encyclopedic/factual tone with a sports quiz or trivia structure, establishing factual cricket statistics about the Indian cricketer Virat Kohli.
> The sentence "This series was one of his best series...the series was remarkable." signals a concluding summary statement wrapping up a FAQ or trivia answer about the 1999 World Cup campaign.
> Final sentence "This series was one of his best series since he debuted" ends a closing remark, prompting a postscript or question elaboration — likely "
> What was his role?" or "During this period?" or "How did Sachin contribute?" or "Was this the most memorable achievement?" or "
> Is this statement correct about these statistics?"

### Claims  (Δ)
- `+0.00120` `THEME/register` The text has an encyclopedic/factual tone
- `+0.00120` `THEME/structure` The text has a sports quiz or trivia structure
- `+0.00120` `THEME/topic` The text establishes factual cricket statistics
- `+0.00120` `ENTITY/person` The text is about the Indian cricketer Virat Kohli
- `+0.00031` `DETAIL/quote` The text contains the sentence "This series was one of his best series...the series was remarkable."
- `+0.00031` `THEME/format` The text contains a concluding summary statement wrapping up a FAQ or trivia answer
- `+0.00031` `ENTITY/event` The text discusses the 1999 World Cup campaign
- `+0.00105` `DETAIL/quote` The text contains the sentence "This series was one of his best series since he debuted"
- `+0.00105` `THEME/format` The text ends with a closing remark

---
## k=6   (intact mse 0.00906)
### AV wrote
> Cricket/sports trivia answer format: informational answer with factual tone, establishing context about Indian cricketer Virat Kohli.
> The sentence "This tournament was pivotal in his career. During this period he played magnificently." signals a concluding remark summarizing the quiz question's statistic or historical achievement.
> Final sentence "This tournament was one of the most memorable phase in his career." ends a concluding statement — expects a closing remark like "What was his role?" or "During this period" or "
> Was this true?" or " Tendulkar also played..." or "In which of these matches?" or "Discuss/list his contribution/statistic" — a factual quiz conclusion about the era.

### Claims  (Δ)
- `+0.00059` `THEME/format` The text is in cricket/sports trivia answer format.
- `+0.00059` `THEME/register` The text has an informational answer with factual tone.
- `+0.00059` `ENTITY/person` The text establishes context about Indian cricketer Virat Kohli.
- `+0.00050` `DETAIL/quote` The text contains the sentence "This tournament was pivotal in his career. During this period he played magnificently."
- `+0.00051` `THEME/structure` The text contains a concluding remark summarizing the quiz question's statistic or historical achievement.
- `+0.00050` `DETAIL/quote` The text contains the final sentence "This tournament was one of the most memorable phase in his career."
- `+0.00068` `THEME/era` The text includes a concluding statement about a factual quiz conclusion about the era.

---
## k=7   (intact mse 0.00923)
### AV wrote
> Encyclopedic/sports commentary format established, with informative tone about a cricket player's statistical achievements in Indian football history.
> The sentence "This series was the peak of his career...This series saw him play exceptionally well." signals a concluding statement or trivia/quiz fact about the mentioned 1982 World Cup highlight.
> Final sentence "This series was the peak of his career" ends a factual closing remark, immediately expecting elaboration like "During this match..." or "
> What role did Sachin Tendulkar play?" or " Tendulkar did he achieve this?" or "Is this correct?" or "In which context was this statement relevant?" — a question or summary of the quiz.

### Claims  (Δ)
- `+0.00069` `THEME/format` The text is in encyclopedic/sports commentary format.
- `+0.00069` `THEME/register` The text has an informative tone.
- `+0.00069` `THEME/topic` The text is about a cricket player's statistical achievements in Indian football history.
- `+0.00002` `DETAIL/quote` The text contains the sentence "This series was the peak of his career...This series saw him play exceptionally well."
- `+0.00002` `ENTITY/event` The text mentions the 1982 World Cup.
- `+0.00283` `DETAIL/quote` The text contains the sentence "This series was the peak of his career" as a factual closing remark.
