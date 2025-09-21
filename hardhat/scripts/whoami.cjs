require("dotenv").config();
const { ethers } = require("ethers");

(async () => {
  const pk = process.env.PRIVATE_KEY;
  const wallet = new ethers.Wallet(pk);
  console.log("Deployer address:", wallet.address);
})();
