
    CHECK PARAMETERS:  WHETHER TO CALCULATE CROSS CORRELATION,  FSKY, NUMBER OF BINS AND THEIR RANGE
    CHECK TRANSFER FUNCTION IN ACCORDANCE WITH THE DATA GENERATOR (if you wish)
    RUN WITH
    CHECK PRIORS OF YOU DO NOT USE STANDARD COSMOLOGY (0.25, 0.05, 0.7)
    run test:
    cobaya-run --test  info_auto.yaml
    and then run cobaya:
    mpirun -n 8 cobaya-run -r  info_auto.yaml >> log.txt &

    sge runs:
    qsub sge.txt
    check that #$ is used instead of # $

        