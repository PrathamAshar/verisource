const fs = require("fs");

async function main() {
  const C = await ethers.getContractFactory("VeriSourceRegistry");
  const c = await C.deploy();
  await c.waitForDeployment();
  const addr = await c.getAddress();
  console.log("Deployed VeriSourceRegistry at:", addr);

  const artifact = await artifacts.readArtifact("VeriSourceRegistry");
  fs.mkdirSync("./abi", { recursive: true });
  fs.writeFileSync("./abi/VeriSourceRegistry.abi.json", JSON.stringify(artifact.abi, null, 2));
}

main().catch((e) => { console.error(e); process.exit(1); });
