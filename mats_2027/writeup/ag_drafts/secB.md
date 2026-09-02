The paper says "AR is only a weak per-claim verifier".

i wanted to investigate if its because of the singal being absent  or buried...(<am i fair in assuming that if we say buried , like it means or refers noise??>)



the signal being the delta(mse). in particular i hypothesise that observed delta of a claim is a sum of underlying delta and noise. by underlying delta i mean, the delta of the claim that we cud calculate in an ideal scenario with 100 percent accracy. 

<i think the equation notation would do better here coz how will someone who is brand new reading this able to understand..although neel is experienced researcher, if i don't convey proper, then there is no point>

now either one of the 2 cases is possible.

1. that the distribution in delta of the underlying for true and false claims is same. -> means they are indistinushable.
2. distribution may be different but the accompanying noise is overwhelming, may be due to the surrounding prose variation across resamples.

sources of confabulation :

1. residual activation vector genuinely encodes it and the AV faithfully read it.
2. AV being a LM in itself made it up.


so. in essence a text judge is primarily blind to residual activation and AV model thought process. the text judge in our case (Haiku 4.5) only receives the passage uptill the position (the prefix) and the claim..and its given 3 choices -> Supported,  or Not in TExt. 

The residual stream activation which served as the bedrock for generation of explanation using AV is available to  only AR for reconstruction in the whole system, so its in. prinicple the only instruemnt that cud tell case 1 from case 2 .
