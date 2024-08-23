# marks-little-lam
LAM (large action model) specification and test simulator.
----
### ORIGIN

In early 2024 Jesse Lyu laid out a vision for a new type of foundational model which he dubbed the "Large Action Model" or "LAM" for short.  He wrote:

>	...LAM‘s modeling approach is rooted in imitation, or learning by demonstration: it observes a human using the interface and aims to reliably replicate the process, even if the interface is presented differently or slightly changed. Instead of having a black-box model uncontrollably outputting actions and adapting to the application during inference...[^1]

The LAM demos hinted at a smart, trainable agent that could independently navigate a website with more than 3 pages without blowing up.  Rabbit sketched out a rough outline of LAM's features, though no concrete definition was put forth.  The Rabbit website goes so far as to envision how it might outperform other foundational models, but stops short of a formal definition.  As of this writing (Aug, 2024), Large Action Model remains a theoretical construct.  In subsequent press releases, Rabbit appears to have abandoned the concept altogether.

Thus we are stepping in to fill this void.

### BUT WHY LAM?

> "I want AI to do my laundry and dishes so that I can do art and writing, not for AI to do my art and writing so that I can do my laundry and dishes"        --Joanna Maciejewska

A major complaint of AI (in its current state) is that it is failing to deliver much tangible ROI.

We posit that this is the result of mal-investment and not an innate shortcoming in any of AI's capabilities.

There are many back-office functions which parasitically tax productivity in any organization, often in not-so-obvious ways.

And while investors have poured multi-billions into GPU fabrication, model training and related activities, the opportunity to refactor the back-office using AI has been largely overlooked -- and frankly starved of investment.

This article is not going to delve into the economics driving this -- perhaps that will be a future discussion.  But SaaS applications are bottom-line driven e.g. *"How much can we reduce headcount by using this SaaS app?"*  Whereas, GenAI solutions (which LAM is) aspire to be top-line driven -- i.e. *"What new revenue and value can be unlocked once employees have been freed from the tyranny of self-service SaaS apps"?*

LAM is the tool to do this.  Unfortunately the required scaffolding to fully implement LAM **_does not exist yet_**.  The purpose of this article (and the associated toolkit code) is to jump-start this initiative.


### NOT A TECHNOLOGY PROBLEM

The conceptual building blocks required to build a LAM  currently exist in rough draft form through a myriad of white papers and sample code -- and possibly a small number of commercial tools.  We are confident that a LAM can be assembled from this assortment of existing tools, models and libraries.  So implementation is not technology constrained.  Rather, the biggest impediments are these:

1.  **WHAT IS A LAM?**  Apart from Rabbit, no one actually knows what a LAM is -- thus a formal definition and software specification is needed.
2.  **HOW DO YOU TEST IT?**  If someone claims to have built a LAM, how would you validate that claim?  Thus a verification methodology to test compliance with the LAM-spec is also required.

This document puts forth a formal software definition for a Large Action Model (LAM-SPEC), while concurrently presenting a suite of benchmarks (LAM-SIM) which can be used by any automnation tool to rigorously assess claims of LAM functionality.

We refer to this collectively as the "LAM-SPEC STANDARD".


### LAM -- A WORKING DEFINITION

What comprises a Large Action Model?  Conceptually, it is an NLP guided inferential agentic framework which can deduce the most likely best next UI action required to satisfy workflow steps in service to a desired goal.  It has the ability to navigate novel (but not unfamiliar) environments using an "environment-blind" approach which primarly leverages visual cognition, waypoints and similarity matching.

Additionally there is an implied notion of "knowledge pooling" of optimal policies compiled by aggregating rewards across multi-user episodes.  I.e. it gets smarter as more users replay more episodes.


### LAM vs. RPA -- WHAT'S THE DIFFERENCE?

Robotic Process Automation (RPA) is a technology that uses software agents or "bots" to automate repetitive, rule-based tasks typically performed by humans. These tasks can include data entry, transaction processing or responding to simple customer service inquiries.  RPA bots interact with software and applications at the system level to execute these tasks quickly and accurately, which tends to improve efficiency and reduce operational costs.

Typically RPA tools are "rules-based" and/or "scripted", meaning they follow a pre-determined set of instructions.  And while they may employ logic (e.g. if condition A is True then execute action B ...) this logic is defined a priori and tends to be deterministic.  A downside of this is the resulting automation scripts tend to be "brittle" and prone to failure when the execution environment mutates in unexpected ways.

In practice, a LAM may perform the exact same actions as an RPA -- but differs in this one important aspect: when confronted with an unfamiliar environment, it will use ANN-type tools to infer the next action(s) to take.

---
**LARGE ACTION MODEL: SOFTWARE SPECIFICATION**
---

What follows is a software requirements specification for LAMs.

##### 1.  Observational Training Modality

	1.1.  A LAM must be able to learn and acquire new skills via lightweight "learn-by-example" observation of an actor.

 		1.1.1.  The LAM shall support a "training mode", which can be activated and deactivated by the user.
		1.1.2.  The LAM shall provide cues (such as visual and audio signalling) when training mode is enabled.
		1.1.3.  When enabled training mode shall "observe" (that is, record) all on-screen activity into a replay buffer.

	1.2. A LAM must recognize GUI elements using an environment-blind or "black-box" paradigm.

		1.2.1.  The LAM shall use image recognition as its primary mode of inspection.
		1.2.2.  The LAM shall have no access to the Document Object Model or any aspect of the Browser's console or API.
		1.2.3.  The LAM shall have no access to system-level events (exluding mouse & keystrokes) or APIs.
		1.2.4.  The LAM shall have no access to third-party APIs.

	1.3.  A LAM shall assign a natural language trigger (identifier) to newly acquired skills.

		1.3.1.  When the LAM acquires a new skill it shall prompt the user to describe the new skill using naming format [<ACTION> <OBJECT>] (e.g. "approve timesheets...")
		1.3.2.  This is to facilitate using natural language to invoke the skill, e.g. "Approve timesheets."

##### 2.  Neural Symbolic Modelling

	2.1.  The LAM shall analyze each training episode in order to create a Neural Symbolic workflow model composed of:

		2.1.1.  A high-level workflow which encodes the training episode.
		2.1.2.  A lattice of UI embeddings for each workflow state transition stored in a vector database.
		2.1.3.  A well defined "happy path" dictionary which maps actions to UI embeddings.
		2.1.4.  A similarity search capability for UI elements
		
		These are described in more detail below.

	2.1.1.  High-level workflow

		- A workflow is a directed sequence of workflow time steps, where each step is a function that takes state s(t) and returns state s(t+1), based on action a(t).
		- Here is a high-level example of a workflow:

			Click the "Login" button --> Click the menu --> Click "Pending" on the menu --> Select (check) the next item (if any) --> Click "Approve" --> Click the "Logout" button

		- These are logged in the replay buffer during training.

	2.1.2.  A lattice of UI embeddings

		- As part of encoding the workflow into the replay buffer the LAM shall de-construct the visual layout of each workflow step as follows:
		- It shall fetch the screenshot corresponding to the current workflow state.
		- It shall perform image recognition on the image, extracting these attributes:
			- container (e.g. top navigation bar, main content area, sidebar, footer)
			- location on page (x,y)
			- size (height, width)
			- category of UI element: buttons, counters, alerts, calendars, icons, etc.
			- color (dominant)
		- It shall perform text recognition (OCR) on the text within each element (i.e the label or titles on the control -- if any)
		- A dictionary shall be created for each UI element containing these attributes as values in a key-value pair.
		- The dictionary shall be converted to embeddings and stored in a vector database.
		- Each element in the lattice shall have a "reward" attribute, with a default value reward=0.
		- The element which was activated during training shall be assigned value reward=1

	2.1.3  A well defined "happy path"

		- The workflow (which leads to the reward scenario) shall be recorded as an ordered list containing these elements:
			- datetime
			- action
			- attributes of UI element which was activated
		- This list and its identifier comprise the replay buffer.

##### 3.  Workflow Editing

	3.1.  The LAM shall be able to articulate a workflow as an ordered sequence of time steps in human-readable format.  

	Here is an example timesheet approval workflow:

		1.  Navigate to URL
		2.  Locate Single Sign On button
		3.  Click the button
		4.  Locate the @ menu
		5.  Hover over the button
		6.  Search for  the Timesheet dropdown
		7.  Hover over the button
		8.  Click the button
		9.  Click the Pending Timesheets
		10. Locate the Dashboard banner
		11. Search for Pending Count
		12. Locate Row 0
		13. Locate Row checkbox
		14. Check Row checkbox
		15. Locate Approve Timesheet button
		16. Search for Pending Count
		17. Locate Avatar
		18. Click Avatar
		19. Click Logout

	3.2  The LAM shall support workflow editing.

		3.2.1 Edits and updates to the LAM script shall be persisted.

##### 4.  Natural Language Activation & Communication

	4.1.  A LAM shall support voice-based natural language prompting as a primary mode of input.

		4.1.1.  A LAM shall support natural language based prompts, either via text-to-speech or via direct text input, 

			e.g. "Approve this week's timesheets"

		4.1.2.  A LAM shall attempt to map a user prompt to a workflow.
		4.1.3.  If a user issued prompt matches a workflow, that workflow shall be initiated.

	4.2  A LAM shall support text-to-speech as a primary mode of output.

		4.2.1.  The LAM shall provide feedback to the user either via text-to-speech or via text, e.g. "I have approved this week's timesheets."

##### 5.  Episodic Replay

	5.1.  A LAM shall be able to replay workflow episodes stored in the replay buffer.
	5.2.  During replay the LAM must be able to fully mimic human GUI gestures and interactions using only mouse and keyboard inputs.
	5.3.  The LAM shall be able to replay the workflow at the same timing cadence logged during training (if so desired)
	5.4. During replay the LAM shall not have access (programmatic or otherwise) to the DOM.


##### 6.  Action-Identification-Evaluation Loop

```
[Initialization]
	|
	v
	[ Action ] ------> [ Identification ]
		|                 	 |
		<-- [ Evaluation ] <---+
			[SUCCESS|FAIL]
```

	6.1.  During replay the LAM shall initialize the environment to a baseline starting State S(0).

		e.g. State S(0) = "Login Page"

	6.2.  Workflow replay shall consist of sequentially executing each workflow time step. 
	6.3.  For each workflow state S(t) LAM shall execute an Action -> Identification -> Evaluation sequence.
	6.4.  The Action step fetches the next action from the replay buffer, e.g. "Click the 'Login' button"

		6.4.1.  Performing Action(t) in State S(t) should cause a transition to State S(t+1)

	6.5.  The Identification step is where the LAM determines which UI element from the UI lattice has the highest probability of  successfully completing the action and moving closer to the overall reward.

	6.6.  The Evaluation step is where the LAM evaluates whether the resultant State matches the Action's expected State(t+1).

		6.6.1.  The evaluation step determines whether the action succeeded or failed.  
		6.6.2.  Retries are permitted.

	6.7.  During workflow replay all the LAM's interactions shall be logged.

##### 7.  Inferential Generalization

	A LAM must be able to generalize to novel scenarios and UI variability is to be expected.  So the LAM shall employ strategies to address this.
	
	7.1.  The LAM shall employ adaptive matching strategies to resolve UI ambiguity.

		7.1.1.  Image recogition should be used.
		7.1.2.  Text recognition should also be used.
		7.1.3.  Similarity scoring (using UI embeddings) should also be used.

##### 8.  Human-in-the-Loop Ambiguity Resolution

	8.1.  When a LAM is unable to resolve the state of its environment it shall notify the user.
	8.2.  When the LAM is unable to find a next suitable action it shall notify the user.
	8.3.  When the LAM has executed all steps in the workflow replay buffer it shall notify the user.

##### 9.  Knowledge Pooling

	9.1.  A LAM shall be able to aggregate learning from multiple users into a consolidated knowledge pool.

---
### LAM-SIM: A LARGE ACTION MODEL SIMULATOR

# <<< COMING SOON >>>

[^1]: https://www.rabbit.tech/research
