from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import Qubit, EPRSocket
import numpy as np
import random


# receive Bell pair from Alice, with a probability that the photon comes from Eve
def receive_bellpair(eprSocket, eavesdroppingProbability):
    local_qubit = eprSocket.recv_keep(1)[0]

    if random.random() < eavesdroppingProbability:
        # simulate eavesdropper by doing a random rotation, thereby making the
        # measurement outcomes uncorrelated with Alice's
        rotangle = random.random() * 2 * np.pi
        local_qubit.rot_Y(angle=rotangle)
    return local_qubit


# pick the measurement bases for all the rounds
def pick_measurement_bases(numBases, conn):
    basis = np.zeros([numBases], dtype=int)
    for k in range(numBases):
        # The E91 protocol uses three measurement bases
        basis[k] = random.randint(0, 2) + 1

        # pick basis with quantum random number generator,
        # seems to make netqsquid unhappy
        # basis[k] = quantum_rnd(3, conn) + 1  # offset measurement bases

    return basis


# generate discrete random numbers from 0 to numOutcomes
def quantum_rnd(numOutcomes, conn):
    numBits = int(np.ceil(np.log2(numOutcomes)))  # number of bits needed to represent numOutcomes

    # powers of two for binary representation
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


# Alice and Bob send their measurement bases to each other
def communicate_measurement_bases(measurementBases, socket):
    # get Alice's basis choices
    measurementBasesAliceString = socket.recv()
    # convert the string back into an array
    measurementBasesAlice = np.array(
        [measurementBasesAliceString[k] for k in range(len(measurementBasesAliceString))], dtype=int)

    # encode Bob's basis choices and send them to Alice
    basisString = ''
    for k in range(len(measurementBases)):
        basisString += str(measurementBases[k])
    socket.send(basisString)

    # save indices where basis choices were the same
    sameBasis = [k for k in range(len(measurementBases)) if (measurementBases[k] == measurementBasesAlice[k])]
    # and the same for different choices
    differentBasis = [k for k in range(len(measurementBases)) if (measurementBases[k] != measurementBasesAlice[k])]

    return measurementBasesAlice, sameBasis, differentBasis


# create a new list of outcomes that only contains the ones where Alice and Bob measured in the same basis
# Bob needs to do a bit flip on his outcomes because they are anti-correlated with Alice's
def filter_outocmes(outcomes, sameBases):
    sameBasisOutcomes = [(outcomes[i]+1) % 2 for i in sameBases]
    return sameBasisOutcomes


# Alice and Bob need to send each other the measurement outcomes that are used to evaluate the CHSH inequality
def communicate_chsh_outcomes(allOutcomes, differentBases, socket):
    if len(differentBases) == 0:  # nothing to communicate
        return

    outcomesAlice = np.zeros([len(differentBases)], dtype=int)
    for k in enumerate(differentBases):
        outcomesAlice[k[0]] = int(socket.recv())
        socket.send(str(allOutcomes[k[1]]))

    return outcomesAlice


# If the channel is noisy Alice and Bob need to perform error correction in order
# to ensure that their keys are identical
def reconcile_errors(outcomes, socket):
    N = int(len(outcomes) - np.mod(len(outcomes), 2))  # number of outcomes truncated to an even number

    payloadAlice = socket.recv_structured().payload
    xorValues = [outcomes[a] ^ outcomes[b] for a, b in zip(payloadAlice[0], payloadAlice[1])]

    # filter out the XOR values that agree
    accept = [int(a == b) for a, b in zip(xorValues, payloadAlice[2])]

    # convert the result to a string and send it back to Alice
    msg = ''.join(str(m) for m in accept)
    socket.send(msg)

    # make a new list containing the outcomes that should be accepted
    correctedOutcomes = [xorValues[i] for i, keep in enumerate(accept) if keep]

    return correctedOutcomes


# Reduce the information that Eve has about the key
# by generating a new key containing the XOR values of pairs of bits in the original key
def perform_privacy_amplification(outcomes, socket):
    # Get the key indices from Alice
    indexPairs = socket.recv_structured().payload

    # perform the XOR operation to generate a shorter, more secure key
    newKey = [outcomes[int(b1)] ^ outcomes[int(b2)] for b1, b2 in zip(indexPairs[0], indexPairs[1])]

    return newKey


# communicate some of the outcomes to estimate the quantum bit error rate (qber)
def estimate_qber(outcomes, socket):
    numOutcomes = len(outcomes)
    numSamples = int(numOutcomes/5)  # use 1/5th of the outcomes to check the qber (could be made a parameter)

    payloadAlice = socket.recv_structured().payload
    outcomesAlice = payloadAlice[0]
    indices = payloadAlice[1]
    outcomesBob = [outcomes[i] for i in indices]
    # find all the errors by doing an XOR between Alice and Bob's outcomes
    # and sum them to get the total amount of bit flips
    # the qber is the number of bit flips divided by the number of bits
    qber = sum(a ^ b for a, b in zip(outcomesAlice, outcomesBob)) / numSamples

    # remove the communicated outcomes from the list of all outcomes
    remainingOutcomes = [x for i, x in enumerate(outcomes) if i not in indices]

    return qber, remainingOutcomes


def main(app_config=None, rounds=500, eavesdroppingProbability=0.05):
    socket = Socket("bob", "alice", log_config=app_config.log_config)
    eprSocket = EPRSocket("alice")
    bob = NetQASMConnection(app_name="bob", log_config=app_config.log_config, epr_sockets=[eprSocket])

    measurementAngles = np.array([0, 45, 90, 135]) * np.pi / 180

    eavesdroppingProbability = 0.1

    with bob:
        # pick measurement bases uniformly randomly
        measurementBases = pick_measurement_bases(rounds, bob)

        allOutcomes = []
        for k in range(rounds):
            qubit = receive_bellpair(eprSocket, eavesdroppingProbability)
            qubit = set_measurement_basis(measurementBases[k], qubit, measurementAngles)
            outcome = qubit.measure()
            bob.flush()

            allOutcomes.append(int(outcome))

        # send / receive measurement bases
        measurementBasesAlice, sameBases, differentBases = communicate_measurement_bases(measurementBases, socket)

        sameBasisOutcomes = filter_outocmes(allOutcomes, sameBases)

        qber, sameBasisOutcomes = estimate_qber(sameBasisOutcomes, socket)

        print(f"Number of measurement in the same basis: {len(sameBases)}")
        print(f"QBER estimate: {qber}")

        outcomesAlice = communicate_chsh_outcomes(allOutcomes, differentBases, socket)

        correctedOutcomes = reconcile_errors(sameBasisOutcomes, socket)

        finalKey = perform_privacy_amplification(correctedOutcomes, socket)
        print(f"Final key: {finalKey}")

        return {"QBER": float(qber)}
