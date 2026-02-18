# Introduction

This repo contains my entry for the Kohonen Self Organising Map Challenge.

## Approach

1. Make code more efficient - Somewhat

1. Make code application friendly - Structuring the code to be easily readible and accessible

1. Think about productionisation -

    1. Quality of life considerations - Comments, seperation of concerns, testing

    1. Developer considerations - Documentation, reproducible environments

    1. Serving - Async background job - will likely be a long-running task that returns result when ready

## Efficiency gains

1. Move away from nested for loops

Libraries such as Numpy and Pytorch offer vectorisation, which allows for quick in-memory vector operations, leading to much better performance over nested for-loops.

Current performance -

1. 10 second training run - 0.244 seconds
1. 1000 second training run - 194.8 seconds

Moving to a python script

1. 10 second training run - 0.205 seconds
1. 1000 second training run - 3.4 seconds

## Questions

- Could the code be made more efficient? A literal interpretation of the instructions above is not necessary.

    > Do away with nested for-loops

- Is the code best structured for later use by other developers and in anticipation of productionisation?

    > No seperation of concerns, not testable, not deployable

- How would you approach productionising this application?

    > Structure as python application, and would serve as docker file.

- Anything else you think is relevant.

    > Code makes an assumption that input data will always be (N x 3) shape. Not explicitly defined.
    > No typing.
    > Greek symbols in code is unusual, and may not render properly in all IDE's.
    > Setting a seed to ensure reproducible results for testing.
    > What if instead of iterating through every input, we could randomly sample from input and approximate a solution? Improved speedup, but lower accuracy
