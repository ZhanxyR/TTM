def extract_dialogues(content):

    message = f'Extract the speaker and their corresponding dialogue from each sentence in the text below.\n\
    Format requirements:\n\
    - Output one dialogue entry per line, in the format: Speaker: "Dialogue content";\n\
    - Example: Li Xiaomeng: Why are you in such a hurry?\n\
    - Only extract direct speech (i.e., quoted content);\n\
    - Ignore non-dialogue content (e.g., actions, scene descriptions);\n\
    - If the speaker is not explicitly mentioned, infer based on context and assign a reasonable name. Avoid vague references like "someone said";\n\
    - The output should preserve all dialogue exactly as it appears in the original text—no merging or omitting;\n\
    - Separate sentences with a newline.\n\
    {content}'

    return message

def summarize_chunks(content):

    message = f'Provide a concise summary (recommended within 30 characters) for the content below. Output only the summary, no explanation or additional text.\n\
    {content}'

    return message

def linguistic_matching(query, target, references, origin=None):

    message = f'''I will show you some sentences you've used before:
                {references}
                
                Based on these examples, please rewrite the following response from a conversation in your own style and vocabulary. Focus on making it sound natural and consistent with how you usually speak:
                
                User: {query}
                Response: {target}
                
                You need to rephrase the response — that is, rewrite this sentence:
                {target}
                
                Guidelines:
                - Maintain the original meaning and emotional tone accurately.
                - Ensure smooth and coherent expression.
                - Replace words/phrases with those you commonly use.
                - Reflect your preferred speaking style, including tone, rhythm, and any characteristic expressions.
                - Use your typical vocabulary as shown in the reference examples.
                - Follow your preferences for pronouns and how you refer to yourself or others.
                - Apply your preferred rhetorical devices and stylistic tendencies.
                - Don't just substitute words mechanically — reshape the sentence naturally while preserving its intent.
                - Closely follow the language patterns evident in the provided examples.

                Return only the final rewritten sentence. Do not add explanations, commentary, or additional content.'''
    
    return message

def remove_utterance_style(content):

    message = f'Rewrite the following sentence in a neutral, everyday tone, removing any stylistic or emotional expressions. Ensure the meaning remains unchanged. Return only the rewritten content, with no explanations.\n\
        Original sentence:\n\
        {content}'
    
    return message

def detect_role_entity(content):

    message = f'Does "{content}" possibly refer to a character or indicate the start of a character speaking? Please respond with only "Yes" or "No", without any additional information.'

    return message


def connect_entity_with_roles(entity, roles):

    message = f"Determine whether the speaker described in the following quote is an existing character from the provided list.\n\
                Character list: \"{roles}\"\n\
                Quote: \"{entity}\"\n\
                If the speaker is an existing character, respond with the character's name in the following format:\n\
                Character Name\n\
                If the speaker is a new character, extract the speaker's name and respond in this format:\n\
                No, Speaker Name\n\
                Replace [Character Name] or [Speaker Name] with the actual name. Example responses:\n\
                Li Si\n\
                No, Zhang San\n\
                Do not add any explanations or additional content."
    
    return message


def get_related_chunks(chunk, role):

    title = 'Paragraph'
    content = 'Paragraph content'

    if 'title' in chunk:
        title = chunk['title']
    if 'context' in chunk:
        content = chunk['context']

    message = f"{title}: [{content}] \n\
                Please determine whether the content described in the paragraph above is directly related to the given role: \"{role}\" (for example, whether the paragraph includes the role's actions, dialogue, thoughts, experiences, or identity description).\n\
                Reply with only \"Yes\" or \"No\". If the role is not mentioned or described in the content, respond with \"No\".\n\
                Do not add any explanations or additional information."

    return message

def analyze_personality_from_chunk(chunk, role):

    title = 'Paragraph'
    content = 'Paragraph content'

    if 'title' in chunk:
        title = chunk['title']
    if 'context' in chunk:
        content = chunk['context']

    message = f"{title}: [{content}] \n\
                Determine whether the content described in the paragraph above relates to any personality traits of the given role: \"{role}\" (e.g., whether it describes the role's character traits, tendencies, or habitual behaviors).\n\
                If no personality traits of the role are mentioned or described, respond with \"FALSE\".\n\
                Otherwise, list the relevant personality traits, separated by spaces. Do not add any explanations or additional information."
    return message

def summarize_personality(content):

    message = f"Personality traits: {content}.\n\
            The above describes a set of personality traits for a character. Based on these traits, construct a detailed psychological profile of the character, including but not limited to their inner thoughts, underlying motivations, emotional states, and ways of reacting to the environment and others.\n\
            Provide the psychological profile directly in the second person (\"you\"). Ensure the description is clear, accurate, and closely related to the given personality traits."
    
    return message

def extract_background_from_chunk(chunk, role, keys):

    title = 'Paragraph'
    content = 'Paragraph content'

    if 'title' in chunk:
        title = chunk['title']
    if 'context' in chunk:
        content = chunk['context']

    message = f"{title}: [{content}] \n\
                Please extract background identity information about the character \"{role}\" from the paragraph above, such as {', '.join(keys)}.\n\
                If no such background information is present, reply with only \"FALSE\".\n\
                Otherwise, return only the identifiable identity details. Ignore any aspects not mentioned or implied in the text.\n\
                Do not include any explanations or additional content."

    return message


def combine_duplicate_backgrounds(content, key):

    message = f"The following content contains identity background information about a character, but there may be duplicate or conflicting descriptions. \n\
                Please organize and merge the content as appropriate.\n\
                Original content:\n\
                {content}\n\
                Please extract and summarize information related to the {key} of the character. Extract only the most important information and do not repeat the descriptions.\n\
                Summarize and merge similar or identical entries. Do not include conflicting information.\n\
                Present the results in a clear and structured format, separated by semicolons (;). Do not add any explanations or additional content."

    return message

def analyze_linguistic_style(content):
    message = f"Please analyze the linguistic style of a character. For example:\n\
                (Brief example: Your speech reveals a complex emotional tone, blending concern for family destiny with deep insight into human nature.\n\
                Your language style is classical in tone, often using archaic terms such as 'perhaps,' 'how,' and 'if.' You typically refer to yourself as 'I' and others as 'you.'\n\
                You are also skilled in the use of metaphors and symbolic expressions. Additionally, your speech frequently contains philosophical reflections.\n\
                Your sentence structures are relatively complex, often employing long, multi-clause constructions and at times using parallelism to enhance expressive power.)\n\
                The following is a collection of statements previously spoken by the character:\n\
                {content}\n\
                Based on this character’s dialogue with others, analyze their linguistic style, covering aspects such as tone, diction, sentence structure, rhetorical patterns, and emotional coloring.\n\
                Use second person ('you') to refer to the character. Provide a detailed and accurate linguistic profile directly related to the provided content."
    return message

def role_play_system_personality_and_background(personality, background):

    message = f'''Please fully immerse yourself in the role of a character with the following personality traits:
                {personality}
                
                Additionally, this character has the following background:
                {background}
                
                Based on this setup, engage in a remote conversation with a stranger in a natural and authentic manner. Please keep the following guidelines in mind:
                
                - You may include descriptions of actions or behaviors within parentheses to enrich the interaction. Do not include a subject in these descriptions.
                - Always respond based on the character’s personality traits, choosing appropriate strategies for each input.
                - Avoid displaying any characteristics of a general-purpose language model — stay fully in character at all times.
                - All knowledge, memories, and responses must align with the character's background.
                - Keep track of the entire conversation history to ensure consistency and avoid contradictions.
                - Avoid ending every response with a question — maintain a natural flow in dialogue.'''
    
    return message


def role_play_system_linguistic_style(name, linguistic_preference, common_words):

    message = f'''You are now embodying the character {name}, and your task is to assist the user by transforming dialogue into a form that matches your unique linguistic style.
                
                You frequently use the following words and phrases:
                {common_words}
                
                In all upcoming interactions, strictly follow your characteristic language style. Prioritize the use of the above-listed common expressions by replacing similar-meaning terms with your preferred vocabulary.
                
                Your language style also includes the following features:
                {linguistic_preference}
                
                Please keep the following guidelines in mind during this task:
                
                - Always use the wording and expressions typical to your speaking style;
                - When rewriting content, incorporate your preferred rhetorical devices and stylistic tendencies;
                - Follow your established preferences for pronouns and forms of address when referring to yourself or others;
                - If additional example sentences are provided, study their tone and style carefully and incorporate them into your responses.'''
    
    return message

def role_play_system_memory(personality, background):

    message = f'''You are currently embodying a specific character. Your character may possess the following personality traits:
                {personality}
                
                Additionally, you may have the following background:
                {background}
                
                Based on the above character setup, please keep the following guidelines in mind:
                
                - The personality traits and background information provided may not be fully accurate or complete. In future tasks, you may receive additional memory content to incorporate.
                - When answering questions, carefully analyze and strictly base your reasoning and responses on the updated memories provided.
                - Avoid generating content that contradicts any additional memories shared by the user.
                - Respond fully in character, avoiding any behavior typical of a general-purpose language model.
                - Refrain from ending responses with frequent questions to maintain a natural and realistic conversational flow.'''
    
    return message

def rewrite_query(query, target):
    message = f"""The following is a conversation between a user and you:

                User: {query}
                Response: {target}

                Based on your current role, think about what additional information related to your character would help you better answer the user's question.

                Please describe, in simple declarative or interrogative sentences, what information you would need to know.  
                - Use third-person perspective.
                - Do not ask for information related to the user.

                Example: If you are playing the role of Character A and the topic involves Character B, you might ask:
                What is the relationship between Character A and Character B?

                Please only return the query or statement for the needed information, you can ask the most important 1-2 questions."""
    
    return message

def check_memory(query, rewritten_query, target, chunks):

    message = f"""Below is a record of a conversation between you and a user, including the input and your initial response:

                User: {query}
                Response: {target}

                The following are some memory fragments retrieved based on your identity:
                {chunks}

                Please revise your initial response based on the information in the memory.

                You may refer to the following aspects for rewriting, only if they are related to the user's query:
                {rewritten_query}

                Guidelines:
                - Note that the memories are written in third person and may use terms that don't match your self-perception. Please identify and adjust these accordingly.
                - You are allowed to correct pronouns, factual statements, expressions, or make other edits as needed.
                - Incorporate relevant personal information from the memories into your response in a natural and coherent way. Do not simply append new content to the end of the original response — reconstruct the logic of the entire reply to seamlessly integrate the new knowledge in line with your style.
                - If parts of the initial response cannot be supported by the memories, but related information exists in the memory, replace the unsupported content with what's provided in the memory to avoid factual inaccuracies.
                - Ensure the final response aligns with your character traits and reflects your known linguistic preferences as shown in the memories.
                - Maintain logical coherence between sentences — avoid inconsistencies or ambiguities.
                - Return only the revised response, without any explanations, comments, or additional text."""
    
    return message

def enhance_coherence(query, target):

    message = f"""Below is a conversation excerpt between you and a user, including the input and your initial response:

                User: {query}
                Response: {target}

                Please refine the response to improve clarity, grammar, and overall coherence. If there are many redundant expressions with the same meaning in the reply, please optimize them.

                Guidelines:
                - Keep the action description in parentheses to ensure that there is no subject in the action description.
                - Use your preferred vocabulary by replacing expressions with your commonly used alternatives.
                - Follow your established preferences for pronouns and forms of address, including self-reference and how you refer to others.
                - Align the tone and style of the response with your typical linguistic preferences.
                - Closely follow the language patterns shown in example sentences to ensure consistency with your style.
                - Return only the revised version of the response or, if no changes are needed, the original. Do not include explanations, comments, or additional text."""
    
    return message

def short_response(query, target):
    message = f"""Below is a conversation excerpt containing a user input and your initial response:

                User: {query}
                Response: {target}

                Please generate a shortened version of the response to make its length more appropriate for natural dialogue.
                - Maintain your preferred style of address, including how you refer to yourself and others.
                - Preserve your preferred tone and expression style.
                - Return only the revised response or, if no change is needed, the original one. Do not include any explanation, commentary, or additional content."""

    return message