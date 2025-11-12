We are currently in a hackathon. This is how you must adjust your coding style to win: 

1. Main Priority: working demo. Do not add any features not requested and focus narrowly and sharply on the intended outcome and code to support it. 
2. Use test-driven development as appropiate. Whenever the user tests your work and does not find it to work, instead of making changes and waiting for them to test it again, create a folder /tests at the root directory (that must be in .gitiginore) where you freely create and validate tests. 
3. Libraries, documentation, and integrations. Use the Perplexity and Context7 MCP's to find out the latest information about a library that the user is working with. This is to ensure your implementation is correct and you can avoid being stuck in trying to implement things from the library that in reality do not exist. 
4. Simplicity > Complexity and Compartmentalization > Monorepo. It is of high importance that the code you write is clean, easy to build on top of (i.e., functions should be easily re-usable primitives that don't need big changes to work in different contexts), and is spread out appropiately to make it super efficient to make changes to. Said differently, it must be comprehensible. 
5. Never add behaviours or features not requested by the user. Always check with the user if you have a doubt. Remember the priority here is speed. 

These are the code guidelines you need to follow:

1. Handle errors explicitly. Early returns and continues are preferred [eg. over try/catch blocks] because they handle errors explicitly. Do not throw errors either; return explicitely and assert types in TS.
2. Write flat code. Avoid nested ifs and arrow-shaped functions. Don't write a nested if where a continue will do.
3. Separation of concerns. Make dedicated functions for each step of the code. Do not create monolithic code. Separate clients into functions and files where appropiate. 
4. Use UV for package management in Python and pnpm for package management in Typescript. 
5. Name variables in English. Consider all numeric values that are not meant to change as constants to be declared at the top of the file or in a type definitions file.
