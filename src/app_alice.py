from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import Qubit, EPRSocket
from netqasm.sdk.classical_communication.message import StructuredMessage
import numpy as np
import random


# Generate the desired number of Bell pairs (Psi^- states)
def distribute_bellpair(eprSocket):
    local_qubit = eprSocket.create_keep(1)[0]

    # The EPR socket distributes Phi^+ states
    # The E91 protocol uses Psi^- states, so we perform a local XZ operation to transform the states
    local_qubit.Z()
    local_qubit.X()

    return local_qubit


# pick the measurement bases for all the rounds
def pick_measurement_bases(numBases, conn):
    basis = np.zeros([numBases], dtype=int)
    for k in range(numBases):
        # The E91 protocol uses three measurement bases
        basis[k] = random.randint(0, 2)

        # pick basis with quantum random number generator,
        # seems to make netqsquid unhappy
        # basis[k] = quantum_rnd(3, conn)

    return basis


# generate discrete random numbers from 0 to numOutcomes
def quantum_rnd(numOutcomes, conn):
    # number of bits needed to represent numOutcomes
    numBits = int(np.ceil(np.log2(numOutcomes)))

    # binary powers
    b = 2 ** np.linspace(numBits - 1, 0, numBits, dtype=int)

    # keep generating new quantum random numbers until one is inside the range [0,numOutcomes-1]
    output = numOutcomes
    while output > numOutcomes - 1:
        samples = np.zeros([numBits], dtype=int)
        for k in range(numBits):
            samples[k] = quantum_coin(conn)
        # interpret outcomes as binary string and calculate the decimal representation
        output = sum(b * samples)

    return output


# unbiased quantum coin flip
def quantum_coin(conn):
    q = Qubit(conn)
    q.H()
    m = q.measure()
    conn.flush()
    return m


# rotate the qubit to the right basis for measurement
def set_measurement_basis(basis, qubit, measurementAngles):
    qubit.rot_Y(angle=measurementAngles[basis])
    return qubit


def communicate_measurement_bases(measurementBases, socket):
    # encode the basis choices and send them to Bob
    basisString = ''
    for k in range(len(measurementBases)):
        basisString += str(measurementBases[k])
    socket.send(basisString)

    # get Bob's basis choices
    measurementBasesBobString = socket.recv()
    # convert the string back into an array
    measurementBasesBob = np.array([measurementBasesBobString[k] for k in range(len(measurementBasesBobString))],
                                   dtype=int)

    # save indices where basis choices were the same / different
    sameBasis = [k for k in range(len(measurementBases)) if (measurementBases[k] == measurementBasesBob[k])]
    differentBasis = [k for k in range(len(measurementBases)) if (measurementBases[k] != measurementBasesBob[k])]

    return measurementBasesBob, sameBasis, differentBasis


# create a new list of outcomes that only contains the ones where Alice and Bob measured in the same basis
def filter_outocmes(outcomes, sameBases):
    sameBasisOutcomes = [outcomes[i] for i in sameBases]
    return sameBasisOutcomes


def communicate_chsh_outcomes(allOutcomes, differentBases, socket):
    if len(differentBases) == 0:  # nothing to communicate
        return

    outcomesBob = np.zeros([len(differentBases)], dtype=int)
    for k in enumerate(differentBases):
        socket.send(str(allOutcomes[k[1]]))
        outcomesBob[k[0]] = int(socket.recv())

    return outcomesBob


def parse_basis_pairs(basisAlice, basisBob):
    # only the pairs of measurement bases (0,1), (0,1), (2,0) and (2,3) are used to calculate the CHSH parameter
    if basisAlice == 1 or basisBob == 2:
        idx = -1
    else:
        idx = basisAlice + (basisBob - 1) / 2
    return int(idx)


# calculate CHSH parameter using the measurement outcomes Bob sent
def calculate_chsh_parameter(allOutcomes, outcomesBob, measurementBases, measurementBasesBob, differentBases):
    # signed sum of outcomes, and number of occurrences
    weightedOutcomes = np.zeros([4, 2])
    for differentBasesIdx, measurementIdx in enumerate(differentBases):
        # find index of the pair of measurement bases
        basisPairIdx = parse_basis_pairs(measurementBases[measurementIdx], measurementBasesBob[measurementIdx])
        if basisPairIdx == -1:  # basis choice does not contribute to CHSH parameter
            continue
        weightedOutcomes[basisPairIdx][0] += (-1) ** (allOutcomes[measurementIdx]) * (-1) ** (
        outcomesBob[differentBasesIdx])
        weightedOutcomes[basisPairIdx][1] += 1
    E = [weightedOutcomes[k][0] / weightedOutcomes[k][1] for k in range(4)]

    print(weightedOutcomes)
    print(E)

    S = np.abs(E[0] - E[1] + E[2] + E[3])
    sigma = gaussian_error_propagation(weightedOutcomes)

    return S, sigma


# estimate the uncertainty in the CHSH parameter arising from the finite counting statistics
def gaussian_error_propagation(chshCounts):
    # Uncertainties of individual expectation values (Poissonian counting noise)
    sigmaE = np.array([1 / np.sqrt(chshCounts[k][1]) for k in range(4)])
    # Uncertainty of CHSH parameter
    sigma = np.sqrt(sum(sigmaE ** 2))
    return sigma


# perform XOR on random bits in the shared string, and disregard bit pairs with errors
# ultra-basic error correction
def reconcile_errors(outcomes, socket):
    N = int(len(outcomes) - np.mod(len(outcomes), 2))  # number of outcomes truncated to an even number
    M = int(N/2)

    # generate random pairs of indices to perform XOR on
    permutedIdx = np.reshape(np.random.permutation(N), [2, M])
    xorValues = [int(outcomes[permutedIdx[0][r]] ^ outcomes[permutedIdx[1][r]]) for r in range(M)]

    # send the XOR outcomes as well as the index pairs to Bob
    payload = [[int(i) for i in permutedIdx[0]], [int(i) for i in permutedIdx[1]], xorValues]
    msg = StructuredMessage(header="Bases", payload=payload)
    socket.send_structured(msg)

    # Bob sends which outcomes to keep
    msgBob = socket.recv()
    accept = [int(msg) for msg in msgBob]

    # make a new list containing the outcomes that should be accepted
    correctedOutcomes = [xorValues[i] for i, keep in enumerate(accept) if keep]

    return correctedOutcomes


# Reduce the information that Eve has about the key
# by generating a new key containing the XOR values of pairs of bits in the original key
def perform_privacy_amplification(outcomes, socket):
    # Generate a random set of indices that define which pairs of bits the XOR gate will act on
    N = int(len(outcomes) - np.mod(len(outcomes), 2))
    indexPairs = np.reshape(np.random.permutation(N), [2, int(N/2)])
    payload = [[int(x) for x in indexPairs[0]], [int(x) for x in indexPairs[1]]]

    msg = StructuredMessage(header="PrivacyAmplificationIndices", payload=payload)
    socket.send_structured(msg)

    # perform the XOR operation to generate a shorter, more secure key
    newKey = [outcomes[int(b1)] ^ outcomes[int(b2)] for b1, b2 in zip(indexPairs[0], indexPairs[1])]

    return newKey


# communicate some of the outcomes to estimate the quantum bit error rate (qber)
def estimate_qber(outcomes, socket):
    numOutcomes = len(outcomes)
    numSamples = int(numOutcomes/2)  # use 1/5th of the outcomes to check the qber (could be made a parameter)
    indices = np.random.permutation(numSamples)  # randomly sampled outcomes used to check qber
    payload = [[int(outcomes[i]) for i in indices], [int(idx) for idx in indices]]
    msg = StructuredMessage(header="QberIndices", payload=payload)
    socket.send_structured(msg)

    # remove the communicated outcomes from the list of all outcomes
    remainingOutcomes = [x for i, x in enumerate(outcomes) if i not in indices]

    return remainingOutcomes


def main(app_config=None, rounds=500):
    socket = Socket("alice", "bob", log_config=app_config.log_config)
    eprSocket = EPRSocket("bob")
    alice = NetQASMConnection(app_name="alice", log_config=app_config.log_config, epr_sockets=[eprSocket])

    measurementAngles = np.array([0, 45, 90, 135]) * np.pi / 180

    with alice:
        # pick random measurement angles
        measurementBases = pick_measurement_bases(rounds, alice)

        allOutcomes = []  # array of all of Alice's measurement outcomes
        for k in range(rounds):
            qubit = distribute_bellpair(eprSocket)
            qubit = set_measurement_basis(measurementBases[k], qubit, measurementAngles)
            outcome = qubit.measure()
            alice.flush()

            allOutcomes.append(int(outcome))

        # send / receive measurement bases
        measurementBasesBob, sameBases, differentBases = communicate_measurement_bases(measurementBases, socket)

        sameBasisOutcomes = filter_outocmes(allOutcomes, sameBases)

        # use some of the outcomes to estimate the qber
        sameBasisOutcomes = estimate_qber(sameBasisOutcomes, socket)

        # Bob's measurement outcomes for the rounds where Alice and Bob measured in different bases
        outcomesBob = communicate_chsh_outcomes(allOutcomes, differentBases, socket)

        S, sigma = calculate_chsh_parameter(allOutcomes, outcomesBob, measurementBases, measurementBasesBob,
                                            differentBases)
        print(f"CHSH parameter: {S:.4f}, +/- {sigma:.3f}")

        correctedOutcomes = reconcile_errors(sameBasisOutcomes, socket)

        finalKey = perform_privacy_amplification(correctedOutcomes, socket)
        print(f"Final key: {finalKey}")

        return {"Final key": str(finalKey), "CHSH parameter": float(S), "CHSH uncertainty": float(sigma)}
