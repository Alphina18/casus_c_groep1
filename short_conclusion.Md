## Conclusion

The model VonBertalanffyGrowth is the best model it got as AICC -95.52737212316885
The model GompertzGrowth is second best model it got a  AICC -75.19514312665639

VonBertalanffyGrowth got less parameter then GompertzGrowth but predict it beter.
So it is a certain winner.

This models are only tested on generated data.
Also this models have almost the same equation so it make sense.

The StochasticGompertzGrowth model is the most random.
This is because it uses the Wiener process with random noise.
This random noice is generated by np.random.normal(0, 1) wich makes it diffrent every time you run it.

Some models are changed with a epsilon of 1e6 to avoid zero division.
This shoudnt impact the models.

The choice of AICC calculation is made cause the data doesnt got a lot of time point.
The program allows actuale data in csv file format.

