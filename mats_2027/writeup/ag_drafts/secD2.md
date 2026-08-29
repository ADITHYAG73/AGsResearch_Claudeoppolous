"AR is only a weak per claim verifier"

main equation -> delta(mse) = mse(abalated explanation) - mse(original explanation)

if delta(mse) > 0 => removed claim worsened the recosntruction on ablated explanation -> claim is load bearing for reconstruction

if delta(mse) < 0 => removed claim helped in reconstruction -> claim is not load bearing for reconstruction

I set out to verify this "claim" (pun unintended) . in fact , as much as this might be a simple sounding statement, it took me a while to register it.The original question that I set out to finad an answer along with my favourite knowledge partner in crime (claude) was this -> does removing the false claims help in reconstruction better? 

(Note this question was directly refered and taken from <i have provide link to neel's google doc here>)

and in the process this is what i found.

<confusion matrix shall come here right? claude??>

about x% of true claims do contribute to a negative delta but they are not necessariyl false claims .

<i feel we got to show some samples or snippets corroborating the above statement>

about y% of false claims do contribute to a positive delta but they are not necessarily true claims either.

<same as above i guess..need samples..even for me to be convinced>

so thinking about this , as mentioned in the oriignial paper also , the expalantions are made by AV which is a language model. the only ever signal for reconstruction is avaialble only for AR and not AV. AV although is injected with a desired position activation in order for us to get a model read out or explaantion at that position, its alanguage model
..am not able to complet this to be honest!!