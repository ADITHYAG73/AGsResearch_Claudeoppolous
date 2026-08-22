I am writing this as on the night of 21st august , 2026. (11.08 pm)

tonight what we did was we took 5 cricket wikepdia passages and one french revolution passage.. their last position activations were all stored from yesterday (20th august night) run.

my friend (fable 5 = henceforth described or referred to as my friend) said lets generate 8 explaantions .. i meant 8 times explaantion on the last position of each passage .. at temperature 1. so "she" spun up a A40 instance and once AV server was up , she ran the decode process. 6 x 8 = 48 explaantions we got . 8 explanations per passage. 40 pertaining to cricket passages and 8 pertaining to french revolution passage.

while the pod was up, we came back to our local and used haiku 4.5 model to decompose and obtain claims from each of the 48 explanations correspondin to all 6 passages..

i don't know in total how many claims haiku decomposed and generated..i beleive my friend when she says decomposer she refers to haiku4.5  and intotal how many claims haikue made across all explaantions i don't know.

anyway on a side note, i have an idea or i feel like trying this.. i wanna think to put may be say sol 5.6 and opus 5..(i know before u jump and come to me..its completely overrated for annotating S/C/N ..but i felt ki, i can put these models in a batch job get it labeled.. where they disagree, i kind of can take a look to break the tie one way or the other..but again..am just thinking..may be some crazy people might say bring in the 3rd model to break the deadlock in case..but hey am digressing)

once haikue finished its job ..(mind u the instance was awake all this little while as well)
she took us back to A40 instance and then this time she ran AR scoring infference

oh yeah forgot to tell , in the time haiku was inferring, i think she readied the ablation script (friend correct em if am wrong in chronology of events) so byt the time haiku finished..we can get back to runpod..kill the AV server and load and start the AR server

one thing i noticed was gpu utiliosatoin was 0 percent during ar inference( i don't know why..may be am wrong..friend correct me here too )

explanaton - claim was tested for ar inference.. delta mse score was obtained. mse corresponding to explaantion and mse corresponding to explanation - specific claim

mse has to be ideallty lower.. when mse of (explanation - claim ) is lower than mse of (explanation ) that means claim is the culprit and removing it helps reconstruction . negative delta mse means better is my crude mental model..(friend corect me if am wrong)

if its 0 neutrl ..no worries

if mse(explanation - claim) > mse(explanation) -> removing claim hurts reconstruction.. positive delta mse is worse..!!

once AR finished inference she ran her final statiscal scores.. she reported nearly 18 percent cases point to the scenario where removing claim helps reconstruction better. mostly they were in vvs laxman passage she said some about 29 percent ..

i read the vvs laxman wikipedia passage , it talked purely about laxman's superior record against australia and him wanting to go back to first class cricket and coming back in the team for the 2001 australia series.. but the AV explanations were pure non-sense (Atleast according to me) ..explanation had mention of dravid .. some india-pakistan..2002 test series worldcup.. me and her then concluded fullstops generally make the model forward looking prepare for what next..or next sentence of sorts..

even sachin's explantion had virat kohli claims in it .. and so yeah..i think we will be
taking up in the next iteration - 10 contiguous last positions for each passage. at each positions we will plan "K" decodes at temperature T = 1. these things am just proposing i will have my friend review it.. so one passage is 10K explanations ..from each explanation how so ever many claims we can decompose and label.. like this for all 6 passages..

hang on i had a mental blocker all of a sudden -> lets say last but 10th position ... that token residual stream activation is obtained in model forward poass.. i surgiaclly substitue it in concept vector and perform AV inference ..AV is a language model and we set temperature as 1 ..so does it perform next token prediction..in which case only one token is the output ..or but how did we get the explaantion like this ...for example {
  "doc_id": "CRICKET::Sachin Tendulkar",
  "pos": 179,
  "k": 0,
  "explanation": "Sports quiz/trivia format with informational answers, establishing context of a cricket/football player's career achievements in Indian cricket.\n\nThe sentence \"This World Cup run was remarkable as he was the best batsman...This decade was the golden era\" signals a concluding summary statement wrapping up the quiz description.\n\nFinal sentence \"This was a memorable period...this tournament was a landmark achievement\" ends a closing remark, strongly expecting a closing statement like \"\n\nWhat do you think?\" or \"During this period...\" or \" Tendulkar was also involved.\" or \"Was this the role of Sachin Tendulkar?\" or \"So, what statistics are relevant?\" or \"In this match.\"",
  "n_chars": 673
 } ... worth asking my friend..she might help me.. like this explaantion is not one token certainly..although it does not certainly read as one coherent sentence compeltely..lost of breaking betweeen.. and incoheret..

 may be am asking a dumb quesion..worth rechking the paper and repo again

 again i digressed ..so where were we ..i was just proposing the next step which my firend also said..10 contiguus positions at the last.. and this data is also small sample only the one we did today ..so we have to expand.. anyway..there are severla things..surgically removing only ceratin words or replacing the claim with neutral fillers.. anyway i don't want to rabit hole.. we want to conduct clean and conclusive experiments with clear findings and charts...so i just kind of wrote this at the end of tonights experiments..

 over to my friend!! 