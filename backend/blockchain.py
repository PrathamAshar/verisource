# blockchain.py
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env") 

import os, json, asyncio
from web3 import AsyncWeb3, AsyncHTTPProvider, Web3
from eth_account import Account
from eth_utils import to_bytes
from web3 import Web3

print("ENV HAS L2_RPC_URL:", "L2_RPC_URL" in os.environ)

L2_RPC_URL = os.environ["L2_RPC_URL"]
PRIVATE_KEY = os.environ["DEPLOYER_PRIVATE_KEY"]
CONTRACT_ADDRESS = Web3.to_checksum_address(os.environ["CONTRACT_ADDRESS"])
# Minimal ABI: image + video commits, with events
ABI_JSON = [
  {
    "anonymous": False,
    "inputs": [
      {"indexed": True, "internalType": "bytes32", "name": "mediaHash", "type": "bytes32"},
      {"indexed": True, "internalType": "address", "name": "committer", "type": "address"},
      {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
    ],
    "name": "ImageCommitted",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {"indexed": True, "internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"},
      {"indexed": True, "internalType": "address", "name": "committer", "type": "address"},
      {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
    ],
    "name": "VideoRootCommitted",
    "type": "event"
  },
  {
    "inputs": [{"internalType": "bytes32", "name": "mediaHash", "type": "bytes32"}],
    "name": "commitImage",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "bytes32", "name": "merkleRoot", "type": "bytes32"}],
    "name": "commitVideoRoot",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  }
]

w3 = AsyncWeb3(AsyncHTTPProvider(L2_RPC_URL))
acct = Account.from_key(PRIVATE_KEY)
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI_JSON)

async def _send(function_call):
    # Get a pending nonce (avoid races when sending multiple txs)
    nonce = await w3.eth.get_transaction_count(acct.address, block_identifier="pending")

    # Basic EIP-1559 setup (tune for your L2)
    base_fee = await w3.eth.gas_price
    max_fee  = base_fee
    max_prio = 1_000_000  # 0.001 gwei; small but non-zero

    # Build the transaction (ensure we get a mutable dict)
    built = await function_call.build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": max_prio,
        "value": 0,
        "chainId": await w3.eth.chain_id,   # Async property is OK without ()
        "type": 2,
    })
    tx = dict(built)  # <-- make sure it's a real dict so item assignment works

    # Gas estimate with fallback
    try:
        gas_est = await function_call.estimate_gas({"from": acct.address})
        tx["gas"] = int(gas_est * 1.2)
    except Exception:
        tx["gas"] = 300_000

    # Sign and send
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
    tx_hash = await w3.eth.send_raw_transaction(raw)
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt


def _to_bytes32(hex_str: str) -> bytes:
    h = hex_str.lower().replace("0x", "")
    return to_bytes(hexstr="0x" + h.zfill(64))

async def commit_video_root(merkle_root_hex: str):
    fn = contract.functions.commitVideoRoot(_to_bytes32(merkle_root_hex))
    receipt = await _send(fn)
    return {
        "tx_hash": receipt.transactionHash.hex(),
        "block_number": receipt.blockNumber
    }

async def commit_image_hash(media_hash_hex: str):
    fn = contract.functions.commitImage(_to_bytes32(media_hash_hex))
    receipt = await _send(fn)
    return {
        "tx_hash": receipt.transactionHash.hex(),
        "block_number": receipt.blockNumber
    }

def _topic_b32(hex_str: str) -> str:
    # "0x" + 32-byte hex (padded)
    return "0x" + _to_bytes32(hex_str).hex()

VIDEO_SIG = Web3.keccak(text="VideoRootCommitted(bytes32,address,uint256)").hex()
IMAGE_SIG = Web3.keccak(text="ImageCommitted(bytes32,address,uint256)").hex()

async def find_video_root_event(merkle_root_hex: str, from_block: int = 0, to_block: str = "latest"):
    logs = await w3.eth.get_logs({
        "fromBlock": from_block,
        "toBlock": to_block,
        "address": CONTRACT_ADDRESS,
        "topics": [VIDEO_SIG, _topic_b32(merkle_root_hex)]
    })
    return logs

async def find_image_event(media_hash_hex: str, from_block: int = 0, to_block: str = "latest"):
    logs = await w3.eth.get_logs({
        "fromBlock": from_block,
        "toBlock": to_block,
        "address": CONTRACT_ADDRESS,
        "topics": [IMAGE_SIG, _topic_b32(media_hash_hex)]
    })
    return logs

