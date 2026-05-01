import numpy as np

refinement_data = np.load('Refinement.npy', allow_pickle=True)
#continuation_data = np.load('PContinuation.npy', allow_pickle=True)

if refinement_data is not None:
    print(f"Array length: {len(refinement_data)}")
    print(refinement_data)

# if continuation_data is not None:
#     print(f"Array length: {len(continuation_data)}")
#     print(continuation_data)