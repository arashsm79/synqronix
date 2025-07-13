import pennylane as qml


def CRZ_layer(nqubits, w):
    """Layer of single-qubit CRZ gates.
    """
    for idx in range(nqubits - 1):
        qml.CRZ(w[idx], wires=[idx, idx + 1])
    qml.CRZ(w[nqubits - 1], wires=[nqubits - 1, 0])  # Connect last qubit to the first one

def CCRZ_layer(thetas):
    """Layer of single-qubit CCRZ gates.
    """
    n = len(thetas)
    if n < 3:
        raise ValueError("CCRZ_layer needs >= 3 qubits (got {}).".format(n))

    for i, theta in enumerate(thetas):
        controls = [i, (i + 1) % n]
        target   = (i + 2) % n
        qml.ctrl(qml.RZ, control=controls)(theta, wires=target)

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)

def gate_layer_2q(gate, params, wire_pairs):
    """
    Apply a two-qubit gate `gate(theta, wires=[i,j])`
    once for every theta in `params`.

    Args
    ----
    gate        – a PennyLane gate class (e.g. qml.CRZ)
    params      – iterable of rotation angles
    wire_pairs  – iterable of (control, target) tuples, same length as params
    """
    for theta, (c, t) in zip(params, wire_pairs):
        gate(theta, wires=[c, t])

def entangling_layer(nqubits):
    """Layers of CZ and RY gates.
    """
    for i in range(0, nqubits - 1):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])

    qml.CNOT(wires=[nqubits-1, 0])


def quantum_net(n_qubits, q_depth, quantum_device):
    
    @qml.qnode(quantum_device, interface='torch')
    def quantum_circuit(inputs, q_weights_flat):
        """
        The variational quantum circuit.
        """

        # Reshape weights
        q_weights = q_weights_flat.reshape(q_depth, 2, n_qubits)

        # Embed features in the quantum node
        qml.AngleEmbedding(inputs, wires=range(
            n_qubits), rotation="Y")
        qml.AngleEmbedding(inputs, wires=range(
            n_qubits), rotation="Z")

        # Sequence of trainable variational layers
        for k in range(q_depth):
            CRZ_layer(n_qubits, q_weights[k][0])
            entangling_layer(n_qubits)
            CCRZ_layer(q_weights[k][1]) # -> need a at least 3 qubits
            
        # Expectation values in the Z basis
        exp_vals = [qml.expval(qml.PauliZ(i))
                    for i in range(n_qubits)]
        return exp_vals

    
    return qml.qnn.TorchLayer(quantum_circuit, {"q_weights_flat": (2*q_depth*n_qubits)})