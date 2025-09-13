from itertools import chain
from math import ceil
import tiktoken


class delta_debug:
    def __init__(self,pruner):
        self.pruner = pruner
        self.prompt = self.pruner.prompt
        

    def tokenization(self, text):
        enc = tiktoken.get_encoding("cl100k_base")
        encodings = enc.encode(text)
        decoded_tokens = []
        for token_id in encodings:
            token_byte = enc.decode_single_token_bytes(token_id)
            token_str = token_byte.decode("utf-8")
            decoded_tokens.append(token_str)
        return decoded_tokens
    

    def line_granularity(self, text):
        return text.split("\n")

    def word_granularity(self, text):
        return text.split()

    def split(self, input, num_chunks):
        chunk_size = ceil(len(input) / num_chunks)
        return [input[i : i + chunk_size] for i in range(0, len(input), chunk_size)]

    def reduce_chunks(self, chunks):

        for i in range(len(chunks)):
            reduced_list = chunks[i]
            reduced_str = ''.join(reduced_list)
            if self.interstingness(reduced_str):
                return reduced_list

        return None

    def complement_chunks(self, chunks):

        for i in range(len(chunks)):
            complement_list = [val for s in range(0, i) for val in chunks[s]]
            complement_list.extend([val for s in range(i + 1, len(chunks)) for val in chunks[s]])
            complement_str = ''.join(complement_list)
            if self.interstingness(complement_str):
                return complement_list
        
        return None

    def delta_debug(self, granularity= 'token'):
        
        if granularity == 'token':
            input = self.tokenization(self.prompt)
        elif granularity == 'line':
            input = self.line_granularity(self.prompt)
        elif granularity == 'word':
            input = self.word_granularity(self.prompt)
        else:
            raise ValueError("Granularity not supported")
        
        num_chunks = 2
        
        while len(input) > 1 :
            
            chunks = self.split(input, num_chunks)
            reduced_input = self.reduce_chunks(chunks)
            
            if reduced_input:
                input = reduced_input
                num_chunks = max(num_chunks - 1, 2)
                continue
            
            complement_input = self.complement_chunks(chunks)

            if complement_input:
                input = complement_input
                num_chunks = max(num_chunks - 1, 2)
                continue

            if num_chunks >= len(input):
                break

            num_chunks = min(num_chunks * 2, len(input))

        input = ''.join(input)
        return input

    def interstingness(self, text):
        return self.pruner.is_similar(text)
    
