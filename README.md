#VeriSource
###PennApps 2025 Submission by Aarav Agarwal, Aditya Agarwal, and Pratham Ashar

## Inspiration

The rise of Generative AI  models have made it more accessible to create fake media, especially images,. As GenAI is constantly improving, it's also becoming more difficult to discern between real and modified/generated images. Having a tool that helps confidently determine whether images were original would make it much harder to spread disinformation.

## What it does

VeriSource records photos taken by a user's device, and generates a unique hash based on the picture to be stored on the blockchain. Therefore, every picture taken from a camera will have its own unique identifier. Users can also upload images to verify if they are real or modified, where image hashes are used to determine if an image is authentic; If there is no match in the database, a neural net is run to try and determine if there were any alterations made (photoshop, cropping) to taken pictures. 

## How we built it

Our app primarily uses python for the backend and node.js for the frontend; It also utilizes MongoDB as a database to store hash receipts, and an Ethereum-blockchain to store image hashes. We first developed a bare-bones backend and frontend, then added blockchain functionality, and finally built our neural net using OpenCV and PyTorch to determine why an image may be inauthentic. 

## Challenges we ran into

We started this project with no idea how the blockchain works, or how to develop using this technology, so the learning curve was our primary challenge. Additionally, getting all the infrastructure set up to accurately leverage its security took some time. 

## Accomplishments that we're proud of

Being successful in our attempt to create unique hashes, and using PyTorch to help determine what alterations happened to an image was a valuable add-on.

## What we learned

We primarily learned how to work with the blockchain, and how hashing actually works. 

## What's next for VeriSource

We hope to scale VeriSource so that it it's image capture functionality is used whenever anyone takes an image for any device, being integrated with camera apps. Additionally, extending its capabilities for videos would make the app more robust.
