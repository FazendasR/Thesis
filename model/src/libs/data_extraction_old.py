import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options


def extract_faculty_and_studyplan(url, search_name, save_directory):
    """
    Extracts text content from the specified search section on the webpage
    and saves it as a .txt file in the specified directory.

    :param url: The webpage URL to scrape.
    :param search_name: The section name to search for (e.g., "Faculty", "Study plan").
    :param save_directory: The directory where the extracted text will be saved.
    """
    
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Setup WebDriver
    driver_path = ChromeDriverManager().install()
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)

    # Retry mechanism for opening the URL
    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            print(f"Attempt {retries + 1}: Accessing {url}")
            driver.get(url)  # Load the webpage
            
            # Wait for elements to load
            wait = WebDriverWait(driver, 10)
            print(f"✅ Successfully accessed {url} for {search_name}")
            break  # Exit loop if successful
        except (TimeoutException, WebDriverException) as e:
            print(f"⚠️ Error on attempt {retries + 1}: {e}")
            retries += 1
            time.sleep(5)  # Wait before retrying

    if retries == max_retries:
        print(f"❌ Failed to load {url} after {max_retries} retries. Exiting function.")
        driver.quit()
        return  # Stop execution if the page couldn't load

    try:
        # Find the main div dynamically
        main_block = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.block.has-aside")))
        print("✅ Found main_block")
        # Find all articles within the main block
        articles = main_block.find_elements(By.CSS_SELECTOR, "article.content__article")
        print(f"✅ Found Articles: {articles}")
        if articles:
            for article in articles:
                try:
                    # Find the <h2> heading inside the article
                    heading_element = article.find_element(By.CSS_SELECTOR, "h2.content__heading.has-border.beta")
                    heading_text = heading_element.text.strip()
                    
                    if search_name in heading_text:
                        print(f"✅ Found Article: {heading_text}")

                        # Extract the corresponding content within <div class="content__article-wrap">
                        content_element = article.find_element(By.CSS_SELECTOR, "div.content__article-wrap")
                        content_html = content_element.get_attribute('outerHTML')

                        # Parse the content using BeautifulSoup
                        soup = BeautifulSoup(content_html, 'html.parser')
                        content_text = soup.get_text(separator=' ', strip=True)
                        print(content_text)

                        # Check for links inside the content and extract the last one
                        links = soup.find_all('a', href=True)
                        if links:
                            last_link = links[-1]['href']  # Get the last link URL
                            print(f"✅ Last Found Link: {last_link}")

                            # Check if the last link ends with one of the desired keywords
                            if last_link.rstrip("/").endswith(("study-plan", "teaching-staff", "faculty")):
                                # Open the linked page
                                driver.get(last_link)
                                time.sleep(5)  # Wait for the page to load

                                if search_name == "Faculty":
                                    try:
                                        section = driver.find_element(By.CSS_SELECTOR, "div.block.has-aside")
                                        print("✅ Faculty link page is not empty. Extracting contect from link page.")

                                        # Extract content using BeautifulSoup
                                        page_soup = BeautifulSoup(driver.page_source, 'html.parser')
                                        section_content = page_soup.find('div', class_='block has-aside')
                                        extracted_text = section_content.get_text(separator=' ', strip=True)

                                    except Exception:
                                        print("⚠️ Faculty link page is empty. ✅ Extracting content from the first page.")
                                        extracted_text = content_text

                                else:
                                    # If search_name is "Study plan", extract the entire content of the linked page
                                    main_element = driver.find_element(By.TAG_NAME, "main")
                                    main_html = main_element.get_attribute('outerHTML')
                                    page_soup = BeautifulSoup(main_html, 'html.parser')

                                    # Remove brochure/country sections if present
                                    unwanted_section = page_soup.find("div", class_="hero-article__info")
                                    if unwanted_section:
                                        unwanted_section.decompose()  

                                    extracted_text = page_soup.get_text(separator=' ', strip=True)
                                    print("✅ Extracting Study plan from the linked page")

                            else:
                                print("Last link is not valid for Faculty or Study Plan. Skipping link.✅ Extracting content from the first page.")
                                extracted_text = content_text

                        else:
                            print("⚠️ No links found in this article. ✅ Extracting content from the first page.")
                            extracted_text = content_text

                        # Extract the last part of the URL for the file name
                        file_name = url.split("/")[-2] + "_" + search_name + ".txt"
                        file_path = os.path.join(save_directory, file_name)

                        with open(file_path, "w", encoding="utf-8") as file:
                            file.write(extracted_text)

                        print(f"✅ Extracted content saved to: {file_path}")
                        break  # Stop after finding the first match for this search_name
                    else: 
                         print(f"⚠️ Not Found Article: {heading_text}")

                except Exception as e:
                    print(f"❌ Skipping article due to error: {e}")

        else:
            print("❌ No articles found inside 'block has-aside'.")

    except Exception as e:
        print(f"❌ Error locating elements: {e}")

    finally:
        # Close WebDriver
        driver.quit()


def extract_faq_section(url, save_directory, search_name="FAQ"):
    """
    Extracts the FAQ section from a given webpage and saves it as 'FAQ_postgraduate_master_degrees.txt'.

    :param url: The webpage URL to scrape.
    :param save_directory: The directory where the extracted text will be saved.
    :param search_name: The section name to search for (default is "FAQ").
    """

    # Setup WebDriver
    driver_path = ChromeDriverManager().install()
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)

    try:
        # Open the target webpage
        driver.get(url)
        wait = WebDriverWait(driver, 10)

        # Find the main block containing articles
        main_block = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "block.has-aside")))
        articles = main_block.find_elements(By.CLASS_NAME, "content__article")

        extracted_text = None

        if articles:
            for article in articles:
                try:
                    # Find the article's <h2> heading
                    heading_element = article.find_element(By.CLASS_NAME, "content__heading")
                    heading_text = heading_element.text.strip()

                    # Extract text only for the specified search section
                    if search_name in heading_text:
                        print(f"✅ Found Article: {heading_text}")

                        # Extract the content within <div class="content__article-wrap">
                        content_element = article.find_element(By.CSS_SELECTOR, "div.content__article-wrap")
                        content_html = content_element.get_attribute('outerHTML')

                        # Parse the content using BeautifulSoup
                        soup = BeautifulSoup(content_html, 'html.parser')

                        # Extract the text from the content
                        extracted_text = soup.get_text(separator=' ', strip=True)
                        print(f"--- Extracted Text ---\n{extracted_text}")

                        break  # Stop after finding the first match
                except Exception as e:
                    print(f"Skipping article due to error: {e}")

        else:
            print("No articles found inside 'block has-aside'.")

        if extracted_text:
            # Ensure save directory exists
            os.makedirs(save_directory, exist_ok=True)

            # Define file name
            file_name = "FAQ_postgraduate_master_degrees.txt"
            file_path = os.path.join(save_directory, file_name)

            # Save extracted text to a file
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(extracted_text)

            print(f"✅ Extracted content saved to: {file_path}")

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Close WebDriver
        driver.quit()

def extract_main_course(url, save_directory):
    '''
    Extracts the MAIN COURSE section from a given webpage and saves it as '(program_name)_main_course.txt'.

    :param url: The webpage URL to scrape.
    :param save_directory: The directory where the extracted text will be saved.
    '''
    # Setup WebDriver
    driver_path = ChromeDriverManager().install()
    service = Service(driver_path)
    options = Options()
    options.headless = False  # Disable headless mode to run with GUI for debugging
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Retry mechanism for loading the page
        retries = 3
        for _ in range(retries):
            try:
                driver.get(url)
                driver.set_page_load_timeout(180)  # Set the maximum time for page load (180 seconds)
                wait = WebDriverWait(driver, 30)  # Increase WebDriverWait to 30 seconds
                print(f"✅ Successfully accessed {url} for Main Content")
                # Wait for the <main> section to be present
                main_element = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/main")))
                break  # Break if the page loaded successfully
            except TimeoutException as e:
                print(f"❌ **Timeout: Page did not load in time. Retrying... ({e})")
                time.sleep(5)  # Wait for 5 seconds before retrying
        else:
            print("❌ **Page load failed after 3 attempts.")
            driver.quit()
            return

        # Extract the last part of the URL for the file name
        file_name = url.split("/")[-2] + "_main_course.txt"
        file_path = os.path.join(save_directory, file_name)

        # Get all <div> elements inside <main>
        div_blocks = main_element.find_elements(By.XPATH, "./div")

        extracted_html = ""

        # Extract content **before** block.has-aside
        for div in div_blocks:
            class_name = div.get_attribute("class")

            if "block has-aside" in class_name:
                break  # Stop before the block.has-aside section

            extracted_html += div.get_attribute("outerHTML") + "\n"

        # Extract block.has-aside content
        has_aside_block = driver.find_element(By.CLASS_NAME, "block.has-aside")
        articles = has_aside_block.find_elements(By.CLASS_NAME, "content__article")

        relevant_headings = ["Program Coordinator", "Length, timetable and exams", "Admissions and fees"]
        extracted_aside_html = ""

        for article in articles:
            try:
                # Extract the heading
                heading_element = article.find_element(By.CLASS_NAME, "content__heading")
                heading_text = heading_element.text.strip()

                if heading_text in relevant_headings:
                    extracted_aside_html += article.get_attribute("outerHTML") + "\n"
            except:
                pass  # Skip if element not found

        # Close WebDriver after extraction
        driver.quit()

        # Parse with BeautifulSoup
        soup_main = BeautifulSoup(extracted_html, "html.parser")
        soup_aside = BeautifulSoup(extracted_aside_html, "html.parser")

        # REMOVE Brochure and Countries sections
        for unwanted_class in ["hero-article__info", "countries-section"]:
            unwanted_section = soup_main.find("div", class_=unwanted_class)
            if unwanted_section:
                unwanted_section.decompose()

        # Extract and clean text
        main_text = soup_main.get_text(separator="\n", strip=True)
        aside_text = soup_aside.get_text(separator="\n", strip=True)

        # Print out relevant progress messages
        print("✅ Main Content Extracted Successfully")
        print(f"✅ {relevant_headings} Extracted Successfully")

        # Save the extracted content to a .txt file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(main_text)
            file.write("\n\n---\n\n")
            file.write(aside_text)

        print(f"✅ Content saved to {file_path}")

    except Exception as e:
        print(f"❌ **Error occurred: {e}")
    finally:
        # Ensure the WebDriver is closed in case of an error
        driver.quit()