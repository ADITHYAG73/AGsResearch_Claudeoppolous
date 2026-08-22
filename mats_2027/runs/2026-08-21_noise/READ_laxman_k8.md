# VVS Laxman — all 8 AV explanations + Haiku claims + Δ
Activation: layer 32, position 254 of 255 = the final `.` after "2001". Same vector, 8 decodes at T=1.
Δ = mse(explanation with that claim's sentence removed) − mse(intact). **Δ>0: claim helped. Δ<0: removal IMPROVED reconstruction.**
## The passage (ground truth for grading)
> Laxman returned to playing first-class cricket in 1999 to regain his place in the national team. In the 1999–2000 season of Ranji Trophy, he broke the record for most runs in a Ranji season when he made 1415 runs, at an average of 108, in just 9 matches notching up eight hundreds – a record that still remains intact. His performance was rewarded when, in January 2000, he was recalled in the Indian squad for the Australian tour. He scored 167 in the third and final Test match at Sydney when the rest of the batsmen struggled to cope with Glenn McGrath's destructive bowling, a rare high point for India in an otherwise disastrous tour. Despite this success against an attack containing both McGrath and Shane Warne, Laxman apparently decided that he would return to domestic cricket, rather than continue playing as opener, a role which he believed did not suit him. As a result, Laxman was out of the Test team for nearly a year. He was recalled in late 2000, and also found a spot in the team for the home series against Australia in 2001.

---
## Resample k=0   (intact mse 0.00946)
### AV wrote
> Encyclopedic/biographical article structure: factual cricket article format, with historical context and narrative summary of a Test match.
> The sentence "The 2001 Test against Australia was the tournament where he made his mark" signals a concluding biographical summary, suggesting a summary of the 2001 series performance.
> Final sentence "The 2001 tournament was where he starred." introduces a concluding statement — likely "This match was..." or "He then played a crucial role in the series" or "The India series saw..." or "This is where he became famous" or "It was in this match that he..." referencing the specific match or turning point.

### Haiku's claims  (Δ when its sentence is removed)
- `-0.00035` `THEME/format` The text is an encyclopedic/biographical article format. **← removal IMPROVED**
- `-0.00035` `THEME/topic` The text is a factual cricket article. **← removal IMPROVED**
- `-0.00035` `THEME/content` The text contains historical context and narrative summary of a Test match. **← removal IMPROVED**
- `+0.00052` `DETAIL/quote` The text contains the sentence "The 2001 Test against Australia was the tournament where he made his mark".
- `+0.00052` `DETAIL/date` The text mentions the year 2001.
- `+0.00052` `ENTITY/event` The text discusses a Test match against Australia.
- `+0.00373` `DETAIL/quote` The text contains the sentence "The 2001 tournament was where he starred."

---
## Resample k=1   (intact mse 0.00943)
### AV wrote
> Encyclopedic/biographical article structure: factual prose about a cricket match, progressing toward a summary of Dravid's Test series performance.
> The sentence "The 2001 series against Australia became the tournament where he made his mark" sets up a concluding remark about the series outcome or landmark innings.
> Final sentence "He participated in the 2001 series where he...was the tournament." signals a concluding biographical statement—likely "This match was his most memorable" or "In this series he showed..." or "It was there that he played the infamous..." or "The India versus England match became..." or "This is where his role..."

### Haiku's claims  (Δ when its sentence is removed)
- `-0.00068` `THEME/format` The text is encyclopedic or biographical article structure **← removal IMPROVED**
- `-0.00068` `THEME/register` The text consists of factual prose **← removal IMPROVED**
- `-0.00068` `THEME/topic` The text is about a cricket match **← removal IMPROVED**
- `-0.00013` `ENTITY/event` The text mentions the 2001 series against Australia
- `-0.00068` `ENTITY/person` The text mentions Dravid **← removal IMPROVED**
- `-0.00013` `DETAIL/quote` The text contains the sentence 'The 2001 series against Australia became the tournament where he made his mark'
- `+0.00440` `DETAIL/quote` The text contains the sentence 'He participated in the 2001 series where he...was the tournament.'
- `-0.00068` `THEME/content` The text discusses Dravid's Test series performance **← removal IMPROVED**

---
## Resample k=2   (intact mse 0.01106)
### AV wrote
> Encyclopedic/biographical article structure: factual cricket article format with historical overview of Indian Test cricketer Javagal Srinath.
> The sentence "The Test series against South Africa in 2007 was the tournament where he made his mark." signals a concluding summary, implying the 2007 series breakthrough.
> Final sentence "He played in the 2007 series where he made his mark." sets up a concluding statement — likely "This match became famous..." or "It was in this series that he..." or "The Indian team's performance in this tournament..." or "He played a crucial role" or "This is where..." referencing the series.

### Haiku's claims  (Δ when its sentence is removed)
- `-0.00055` `THEME/format` The text has an encyclopedic/biographical article structure **← removal IMPROVED**
- `-0.00055` `THEME/format` The text is a factual cricket article **← removal IMPROVED**
- `-0.00055` `THEME/topic` The text contains a historical overview of Indian Test cricketer Javagal Srinath **← removal IMPROVED**
- `-0.00055` `ENTITY/person` The text mentions Indian Test cricketer Javagal Srinath **← removal IMPROVED**
- `+0.00056` `DETAIL/quote` The text contains the sentence "The Test series against South Africa in 2007 was the tournament where he made his mark."
- `+0.00056` `ENTITY/event` The text discusses a Test series against South Africa in 2007
- `+0.00056` `DETAIL/date` The text mentions the year 2007
- `+0.00239` `DETAIL/quote` The text contains the sentence "He played in the 2007 series where he made his mark."
- `+0.00239` `ENTITY/team` The text mentions the Indian team

---
## Resample k=3   (intact mse 0.01014)
### AV wrote
> Encyclopedic/biographical article structure: article follows a factual cricket history format, with a summary of Dravid's Test series performance.
> The sentence "The 2002 India tour was the venue where he played the match against Australia." signals a concluding biographical summary, suggesting a legacy statement about the series.
> Final sentence "The 2002 series became the venue where he played the match." ends a summary clause, immediately expecting elaboration like "This series was his breakthrough" or "In that series he..." or "The match against Australia is where he showed..." or "This is where he played..." or "He then became important."

### Haiku's claims  (Δ when its sentence is removed)
- `-0.00041` `THEME/structure` The text follows an encyclopedic/biographical article structure. **← removal IMPROVED**
- `-0.00041` `THEME/format` The text follows a factual cricket history format. **← removal IMPROVED**
- `+0.00022` `DETAIL/quote` The text contains the sentence 'The 2002 India tour was the venue where he played the match against Australia.'
- `-0.00041` `ENTITY/event` The text mentions a Test series performance. **← removal IMPROVED**
- `-0.00041` `ENTITY/person` The text mentions Dravid. **← removal IMPROVED**
- `+0.00022` `ENTITY/event` The text mentions a 2002 India tour.
- `+0.00022` `ENTITY/team` The text mentions Australia.
- `+0.00673` `DETAIL/quote` The text contains a final sentence 'The 2002 series became the venue where he played the match.'
- `+0.00022` `DETAIL/date` The text mentions 2002.

---
## Resample k=4   (intact mse 0.01133)
### AV wrote
> Encyclopedic/biographical article structure: the text follows a factual cricket article format, with a summary of the India-West Indies Test series.
> The sentence "The Test series against the West Indies in 2002 was the venue where he made his breakthrough" signals a concluding biographical summary about Dravid.
> Final sentence "The tournament held in 2002 was the venue where he made his breakthrough." sets up a concluding statement — likely "This series was..." or "It was in this series that..." or "He played a pivotal role in the series" or "The match against the 2003 ODI World Cup."

### Haiku's claims  (Δ when its sentence is removed)
- `-0.00114` `THEME/format` The text follows a factual cricket article format **← removal IMPROVED**
- `-0.00114` `THEME/topic` The text is about a summary of the India-West Indies Test series **← removal IMPROVED**
- `+0.00011` `ENTITY/event` The text mentions a Test series against the West Indies in 2002
- `+0.00011` `ENTITY/person` The text discusses Dravid
- `+0.00011` `DETAIL/quote` The text contains the sentence "The Test series against the West Indies in 2002 was the venue where he made his breakthrough"
- `+0.00291` `DETAIL/quote` The text contains the sentence "The tournament held in 2002 was the venue where he made his breakthrough."
- `+0.00011` `DETAIL/date` The text mentions the year 2002

---
## Resample k=5   (intact mse 0.01034)
### AV wrote
> Encyclopedic/biographical article structure: the text follows a structured cricket biography format, presenting factual details about a specific Indian Test match.
> The sentence "The 2002 Test series against Australia was the venue where he made his breakthrough" signals a concluding summary about the Chennai Test.
> Final sentence "He played in the 2002 Test series against Australia." ends a biographical conclusion clause, strongly expecting elaboration — likely "This match was..." or "In this series he..." or "It was there that he excelled..." or "The Indian team series became his most memorable performance" or "This is where he showed..." referencing the controversial match details.

### Haiku's claims  (Δ when its sentence is removed)
- `-0.00087` `THEME/structure` The text follows a structured cricket biography format. **← removal IMPROVED**
- `-0.00087` `THEME/content` The text presents factual details about a specific Indian Test match. **← removal IMPROVED**
- `+0.00012` `ENTITY/event` The text mentions the 2002 Test series against Australia.
- `+0.00012` `DETAIL/quote` The text contains the sentence "The 2002 Test series against Australia was the venue where he made his breakthrough".
- `+0.00012` `ENTITY/place` The text references the Chennai Test.
- `+0.00306` `DETAIL/quote` The text contains the sentence "He played in the 2002 Test series against Australia."
- `+0.00306` `THEME/structure` The text ends with a biographical conclusion clause.
- `+0.00306` `THEME/content` The text discusses controversial match details.
- `  (unmatched)` `THEME/content` The text presents biographical information about an individual player.

---
## Resample k=6   (intact mse 0.00926)
### AV wrote
> Encyclopedic/biographical article structure: the text follows a Wikipedia format, presenting factual cricket history with a detailed narrative of the India vs Pakistan series.
> The sentence "The Test series against Pakistan in the 2002 World Cup was where he made his mark" signals a concluding summary statement about Dravid's performance.
> Final sentence "The tournament continued from the 2002 World Cup." ends a biographical footnote establishing the series context — immediately expects elaboration like "This match was where he..." or "
> It was in this series that..." or "The series proved to be his most memorable" or "He played a crucial role in that series."

### Haiku's claims  (Δ when its sentence is removed)
- `+0.00006` `THEME/structure` The text follows a Wikipedia format.
- `+0.00006` `THEME/topic` The text presents factual cricket history.
- `+0.00006` `THEME/topic` The text contains a detailed narrative of the India vs Pakistan series.
- `+0.00100` `DETAIL/quote` The text contains the sentence 'The Test series against Pakistan in the 2002 World Cup was where he made his mark'.
- `+0.00100` `ENTITY/person` The text mentions Dravid.
- `+0.00424` `DETAIL/quote` The text contains the final sentence 'The tournament continued from the 2002 World Cup.'
- `+0.00100` `ENTITY/event` The text discusses the 2002 World Cup.
- `+0.00100` `ENTITY/event` The text discusses a Test series against Pakistan.

---
## Resample k=7   (intact mse 0.00993)
### AV wrote
> Encyclopedic/biographical article structure: article format with factual tone has covered Tendulkar's Test series history and the India vs West Indies match.
> The sentence "The 2002 series against the West Indies was where he made his mark" signals a concluding summary statement, resolving the narrative.
> Final sentence "He played the 2002 series against the West Indies." ends a biographical summary — immediately expects elaboration like "This series was his most memorable" or "In this tournament..." or "It was there that he..." or "The match against Pakistan became the turning point" or "This series is where he showed..." citing the specific innings or role.

### Haiku's claims  (Δ when its sentence is removed)
- `-0.00110` `THEME/format` The text has an encyclopedic/biographical article format **← removal IMPROVED**
- `-0.00110` `THEME/register` The text has a factual tone **← removal IMPROVED**
- `-0.00110` `THEME/topic` The text covers Tendulkar's Test series history **← removal IMPROVED**
- `-0.00110` `ENTITY/event` The text mentions the India vs West Indies match **← removal IMPROVED**
- `+0.00076` `DETAIL/quote` The text contains the sentence 'The 2002 series against the West Indies was where he made his mark'
- `+0.00483` `DETAIL/quote` The text contains the sentence 'He played the 2002 series against the West Indies.'
- `+0.00076` `ENTITY/date` The text mentions 2002
