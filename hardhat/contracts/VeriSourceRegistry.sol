// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract VeriSourceRegistry {
    event ImageCommitted(bytes32 indexed mediaHash, address indexed committer, uint256 timestamp);
    event VideoRootCommitted(bytes32 indexed merkleRoot, address indexed committer, uint256 timestamp);

    function commitImage(bytes32 mediaHash) external {
        emit ImageCommitted(mediaHash, msg.sender, block.timestamp);
    }
    function commitVideoRoot(bytes32 merkleRoot) external {
        emit VideoRootCommitted(merkleRoot, msg.sender, block.timestamp);
    }
}
