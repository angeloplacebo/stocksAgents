# IMPORT DAS LIBS
# import json
import os
from datetime import datetime

import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st


def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock


yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="""
    Fetches stock prices for {ticket} from the last year about
    a specific stock from Yahoo Finance API""",
    func=lambda ticket: fetch_stock_price(ticket)
  )


# IMPORTANDO OPENAI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")


stockPriceAnalyst = Agent(
  role="Senior stock price Analyst",
  goal="Find the {ticket} stock price and analyses trends",
  backstory="""
  You're a highly experienced in analyzing the price of
  an specific stock and make predictions about its future price.""",
  llm=llm,
  verbose=True,
  max_iter=5,
  memory=True,
  allow_delegation=False,
  tools=[yahoo_finance_tool]
)


getStockPrice = Task(
  description="""Analyze the stock {ticket} price history
  and create a trend analyses of up, down or sideways""",
  expected_output=""" Specify the current trend stock price
  - up, down or sideways.
  eg. stock = 'AAPL, price UP'
  """,
  agent=stockPriceAnalyst
)


# IMPORTANDO A TOOL DE SEARCH
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


newsAnalyst = Agent(
  role="Stock News Analyst",
  goal="""Create a shot summary of the market news
  related to the stock {ticket} company.
  Specify the current trend - up, down or sideways with
  the news context. For each request stock asset,
  specify a number between 0 and 100,
  where 0 is extreme fear and 100 extreme greed.""",
  backstory="""
  You're highly expreienced in analyzing the market trends and news
  and have tracked assets for more the 10 years.

  You're also master level analyst in the tradicional markets
  and have deep understanding of human psychology.

  You understand new, theirs tittles and information,
  but you look at those with a health dose of skepticism.
  You consider also the source of the news articles.
  """,
  llm=llm,
  verbose=True,
  max_iter=10,
  memory=True,
  allow_delegation=False,
  tools=[search_tool]
)


# In[99]:


get_news = Task(
  description=f"""Take the stock and always include BTC to it (if not request).
  use the search tool to search each one individualy.

  The current date is {datetime.now()}

  Compose the results into a helpful report""",
  expected_output="""A summary of the overall market
  and one sentece summary for each request asset.
  Include a fear/greed score for each asset based on the news. Use the format:
  <STOCK ASSET>
  <SUMMARY BASED ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED SCORE>
  """,
  agent=newsAnalyst
)


# In[100]:


stockAnalystWrite = Agent(
  role="Senior Stock Analyst Writer",
  goal="""Analyze the trends price and news and write an insightful compelling
  and informative 3 paragraph long newsletter
  bases on the stock report and price trend. """,
  backstory="""You're widely accdepted as the best stock analyst in the market.
  You undestand comples concepts and create compelling stories
  and narratives that resonate with wider audiences.

  You undestand macro factors and combine multiple theories
  - eg. cycle theory and fundamental analyses.
  You're able to hold multiple opinions when analyzing anything.
  """,
  llm=llm,
  verbose=True,
  max_iter=5,
  memory=True,
  allow_delegation=True
)


# In[101]:


writeAnalyses = Task(
  description="""Use the stock price trend and the stock news report to create
  an analyses and write the newsletter about the {ticket} company
  that is brief and highlights the most important points.
  Foucus on the stock price trend, news and fear/greed score.
  What are the near future considerations?
  Include the previous analyses of stock trend and news summary.
  """,
  expected_output="""An eloquent 3 paragraphs newsletter
  formated as markdown in an easy readable manner. It should contain:

  - 3 bullets executive summary
  - Introduction - set the overall picture and spike up the interest
  - main part provides the meat of the analysis
    including the news summary and fear/greed scores
  - summary
    - key facts and concrete future trend prediction
    - up, down sideways.
  """,
  agent=stockAnalystWrite,
  context=[getStockPrice, get_news]
)


# In[102]:


crew = Crew(
  agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
  tasks=[getStockPrice, get_news, writeAnalyses],
  verbose=True,
  process=Process.hierarchical,
  full_output=True,
  share_crew=False,
  manager_llm=llm,
  max_iter=15
)


# results = crew.kickoff(inputs={'ticket': 'AAPL'})

with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not topic:
        st.error("Please enter a topic to research")
    else:
        results = crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of your research:")
        st.write(results['final_output'])
