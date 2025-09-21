require("dotenv").config();
const { ethers } = require("ethers");

(async () => {
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const addr = new ethers.Wallet(process.env.PRIVATE_KEY).address;
  const bal = await provider.getBalance(addr);
  console.log("Address:", addr);
  console.log("Balance (ETH):", ethers.formatEther(bal));
})();
