"""
Sample Document Generator
Creates sample legal documents for testing the Hierarchical RAG system
"""

import os
from datetime import datetime, timedelta


def create_sample_service_agreement():
    """Create a sample service agreement"""
    
    content = """
SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 15, 2024 ("Effective Date"), 
by and between TechCorp Solutions LLC, a Delaware corporation with its principal place of 
business at 123 Tech Street, San Francisco, CA 94102 ("Provider"), and Global Innovations Inc., 
a California corporation with its principal place of business at 456 Innovation Ave, 
Los Angeles, CA 90001 ("Client").

ARTICLE I - DEFINITIONS

1.1 Services
"Services" means the software development and consulting services to be provided by Provider 
to Client as described in Exhibit A attached hereto.

1.2 Deliverables
"Deliverables" means the work product, materials, and documentation to be delivered by 
Provider to Client as part of the Services.

1.3 Proprietary Information
"Proprietary Information" means all confidential and proprietary information disclosed by 
either party to the other party.

ARTICLE II - SCOPE OF SERVICES

2.1 Service Description
Provider agrees to perform the Services described in Exhibit A in a professional and 
workmanlike manner in accordance with industry standards.

2.2 Timeline
The Services shall commence on February 1, 2024, and shall continue for a period of 
twelve (12) months, unless earlier terminated in accordance with Section 7.

2.3 Deliverables Schedule
Provider shall deliver the Deliverables according to the timeline set forth in Exhibit B.

ARTICLE III - PAYMENT TERMS

3.1 Fees
Client shall pay Provider a total fee of $250,000 for the Services ("Fees"), payable as follows:
(a) $75,000 upon execution of this Agreement
(b) $100,000 upon completion of Phase 1 Deliverables
(c) $75,000 upon completion of Final Deliverables

3.2 Payment Schedule
All payments are due within thirty (30) days of invoice date. Invoices will be sent by Provider 
upon completion of each milestone.

3.3 Late Payment
Late payments shall accrue interest at the rate of five percent (5%) per annum, or the 
maximum rate permitted by law, whichever is less.

3.4 Expenses
Client shall reimburse Provider for all reasonable and pre-approved out-of-pocket expenses 
incurred in connection with the Services, not to exceed $10,000 without prior written approval.

ARTICLE IV - INTELLECTUAL PROPERTY RIGHTS

4.1 Ownership of Deliverables
Upon full payment of all Fees, Client shall own all right, title, and interest in and to 
the Deliverables, including all intellectual property rights therein.

4.2 License to Provider Tools
Provider hereby grants Client a perpetual, non-exclusive, royalty-free license to use any 
pre-existing tools, templates, or methodologies incorporated into the Deliverables.

4.3 Reservation of Rights
Provider retains all rights to its pre-existing intellectual property, proprietary methods, 
and know-how developed independently of this Agreement.

ARTICLE V - CONFIDENTIALITY

5.1 Confidential Information
Each party agrees to maintain the confidentiality of all Proprietary Information received 
from the other party and to use such information solely for purposes of this Agreement.

5.2 Duration of Obligations
The confidentiality obligations under this Article shall survive for a period of three (3) 
years following the termination or expiration of this Agreement.

5.3 Exceptions
Confidentiality obligations shall not apply to information that: (a) is publicly available, 
(b) is independently developed, or (c) is required to be disclosed by law.

ARTICLE VI - WARRANTIES AND REPRESENTATIONS

6.1 Provider Warranties
Provider warrants that:
(a) It has the right and authority to enter into this Agreement
(b) The Services will be performed in a professional manner
(c) The Deliverables will be free from material defects for ninety (90) days

6.2 Client Warranties
Client warrants that it has the authority to enter into this Agreement and to grant the 
rights granted herein.

6.3 Disclaimer
EXCEPT AS EXPRESSLY PROVIDED HEREIN, PROVIDER MAKES NO OTHER WARRANTIES, EXPRESS OR IMPLIED, 
INCLUDING WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.

ARTICLE VII - TERM AND TERMINATION

7.1 Term
This Agreement shall commence on the Effective Date and continue for twelve (12) months 
unless earlier terminated as provided herein.

7.2 Termination for Convenience
Either party may terminate this Agreement for any reason upon sixty (60) days prior 
written notice to the other party.

7.3 Termination for Cause
Either party may terminate this Agreement immediately upon written notice if the other 
party materially breaches this Agreement and fails to cure such breach within thirty (30) 
days of receiving written notice.

7.4 Effect of Termination
Upon termination, Client shall pay Provider for all Services performed and expenses 
incurred up to the date of termination. Provider shall deliver all completed Deliverables 
and work in progress to Client.

ARTICLE VIII - LIMITATION OF LIABILITY

8.1 Cap on Liability
EXCEPT FOR BREACHES OF CONFIDENTIALITY OR INTELLECTUAL PROPERTY INFRINGEMENT, NEITHER 
PARTY'S LIABILITY SHALL EXCEED THE TOTAL FEES PAID OR PAYABLE UNDER THIS AGREEMENT.

8.2 Consequential Damages
IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR 
CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS.

ARTICLE IX - INDEMNIFICATION

9.1 Provider Indemnification
Provider shall indemnify and hold harmless Client from any claims arising from Provider's 
gross negligence or willful misconduct in performing the Services.

9.2 Client Indemnification
Client shall indemnify and hold harmless Provider from any claims arising from Client's 
use of the Deliverables in a manner not authorized under this Agreement.

ARTICLE X - GENERAL PROVISIONS

10.1 Governing Law
This Agreement shall be governed by the laws of the State of California, without regard 
to conflict of law principles.

10.2 Dispute Resolution
Any disputes arising under this Agreement shall first be subject to good faith negotiation. 
If not resolved within thirty (30) days, disputes shall be resolved by binding arbitration 
in San Francisco, California.

10.3 Assignment
Neither party may assign this Agreement without the prior written consent of the other party, 
except that either party may assign to a successor in connection with a merger or acquisition.

10.4 Entire Agreement
This Agreement constitutes the entire agreement between the parties and supersedes all 
prior agreements and understandings, whether written or oral.

10.5 Amendment
This Agreement may only be amended by written instrument signed by both parties.

10.6 Severability
If any provision of this Agreement is found to be invalid or unenforceable, the remaining 
provisions shall remain in full force and effect.

10.7 Force Majeure
Neither party shall be liable for any failure to perform due to causes beyond its reasonable 
control, including acts of God, war, riot, or natural disasters.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.

TECHCORP SOLUTIONS LLC                  GLOBAL INNOVATIONS INC.

By: _________________________          By: _________________________
Name: John Smith                       Name: Sarah Johnson
Title: CEO                             Title: VP of Technology
Date: January 15, 2024                Date: January 15, 2024
"""
    
    return content


def create_sample_nda():
    """Create a sample Non-Disclosure Agreement"""
    
    content = """
NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is made effective as of March 1, 2024, between 
Alpha Ventures LLC ("Disclosing Party") and Beta Technologies Inc. ("Receiving Party").

RECITALS

WHEREAS, the parties wish to explore a potential business relationship;
WHEREAS, in connection with such discussions, Disclosing Party may disclose certain 
confidential and proprietary information to Receiving Party;
NOW, THEREFORE, in consideration of the mutual covenants and agreements herein contained, 
the parties agree as follows:

ARTICLE I - DEFINITIONS

1.1 Confidential Information
"Confidential Information" means any and all technical and non-technical information disclosed 
by Disclosing Party, including but not limited to:
(a) Trade secrets, inventions, algorithms, software code
(b) Business plans, financial information, customer data
(c) Marketing strategies, product roadmaps
(d) Any information marked as "Confidential" or that reasonably should be considered confidential

1.2 Purpose
"Purpose" means evaluating and potentially establishing a business partnership between the parties.

ARTICLE II - CONFIDENTIALITY OBLIGATIONS

2.1 Non-Disclosure
Receiving Party agrees not to disclose any Confidential Information to any third party without 
the prior written consent of Disclosing Party.

2.2 Use Restrictions
Receiving Party shall use Confidential Information solely for the Purpose and for no other purpose.

2.3 Protection Measures
Receiving Party shall protect Confidential Information using the same degree of care it uses 
to protect its own confidential information, but in no event less than reasonable care.

2.4 Limited Disclosure
Receiving Party may disclose Confidential Information only to its employees, consultants, or 
advisors who have a legitimate need to know and who are bound by confidentiality obligations 
at least as restrictive as those contained herein.

ARTICLE III - EXCEPTIONS

3.1 Excluded Information
The obligations under Article II shall not apply to information that:
(a) Is or becomes publicly available through no breach of this Agreement
(b) Was rightfully in Receiving Party's possession prior to disclosure
(c) Is independently developed by Receiving Party without use of Confidential Information
(d) Is rightfully received from a third party without confidentiality obligations
(e) Is required to be disclosed by law or court order

3.2 Required Disclosure
If Receiving Party is required by law to disclose Confidential Information, it shall provide 
prompt notice to Disclosing Party to enable Disclosing Party to seek a protective order.

ARTICLE IV - TERM AND RETURN OF MATERIALS

4.1 Term
This Agreement shall commence on the effective date and continue for a period of five (5) years.

4.2 Survival
The confidentiality obligations shall survive termination of this Agreement and continue for 
five (5) years from the date of disclosure of each item of Confidential Information.

4.3 Return of Materials
Upon termination or upon request by Disclosing Party, Receiving Party shall promptly return 
or destroy all Confidential Information and certify in writing such return or destruction.

ARTICLE V - NO LICENSE

5.1 Ownership
All Confidential Information remains the sole property of Disclosing Party. No license or 
other rights in Confidential Information are granted except as expressly set forth herein.

5.2 No Obligation
Nothing in this Agreement obligates either party to proceed with any transaction or business 
relationship.

ARTICLE VI - REMEDIES

6.1 Injunctive Relief
Receiving Party acknowledges that any breach of this Agreement may cause irreparable harm 
for which monetary damages would be inadequate. Disclosing Party shall be entitled to seek 
equitable relief, including injunction and specific performance.

6.2 Attorneys' Fees
The prevailing party in any action to enforce this Agreement shall be entitled to recover 
reasonable attorneys' fees and costs.

ARTICLE VII - GENERAL PROVISIONS

7.1 Governing Law
This Agreement shall be governed by the laws of the State of Delaware.

7.2 Jurisdiction
The parties consent to the exclusive jurisdiction of courts in Wilmington, Delaware.

7.3 Entire Agreement
This Agreement constitutes the entire agreement regarding confidentiality and supersedes 
all prior agreements.

7.4 Amendment
This Agreement may be amended only by written instrument signed by both parties.

7.5 Waiver
No waiver of any provision shall be deemed a waiver of any other provision or subsequent breach.

7.6 Counterparts
This Agreement may be executed in counterparts, each of which shall be deemed an original.

IN WITNESS WHEREOF, the parties have executed this Agreement.

ALPHA VENTURES LLC                      BETA TECHNOLOGIES INC.

By: _________________________          By: _________________________
Name: Michael Chen                     Name: Rebecca Martinez  
Title: CEO                             Title: CTO
Date: March 1, 2024                   Date: March 1, 2024
"""
    
    return content


def create_sample_employment_contract():
    """Create a sample employment contract"""
    
    content = """
EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into as of April 1, 2024, by and between 
DataFlow Industries Inc., a New York corporation ("Company"), and Jennifer Williams ("Employee").

ARTICLE I - EMPLOYMENT

1.1 Position and Duties
Company hereby employs Employee as Senior Software Engineer. Employee shall report to the 
VP of Engineering and shall perform such duties as are customarily associated with such position.

1.2 Best Efforts
Employee agrees to devote her best efforts and full business time to the performance of her duties.

1.3 Location
Employee's primary work location shall be Company's office at 789 Tech Plaza, New York, NY 10001, 
with the option to work remotely up to two (2) days per week.

ARTICLE II - COMPENSATION AND BENEFITS

2.1 Base Salary
Company shall pay Employee an annual base salary of $180,000, payable in accordance with 
Company's standard payroll practices.

2.2 Annual Bonus
Employee shall be eligible for an annual performance bonus of up to thirty percent (30%) 
of base salary, based on achievement of mutually agreed performance objectives.

2.3 Equity Compensation
Employee shall receive stock options to purchase 20,000 shares of Company common stock, 
vesting over four (4) years with a one-year cliff.

2.4 Benefits
Employee shall be entitled to participate in all employee benefit plans offered by Company, 
including:
(a) Health, dental, and vision insurance
(b) 401(k) retirement plan with Company matching
(c) Life and disability insurance
(d) Paid time off: 20 days annually

2.5 Expenses
Company shall reimburse Employee for all reasonable business expenses incurred in performing 
her duties, subject to Company's expense reimbursement policy.

ARTICLE III - TERM AND TERMINATION

3.1 Term
This Agreement shall commence on April 1, 2024, and continue until terminated by either party 
as provided herein.

3.2 At-Will Employment
Employee's employment is at-will and may be terminated by either party at any time, with or 
without cause, subject to the notice provisions below.

3.3 Notice of Termination
Either party may terminate employment by providing thirty (30) days written notice, provided 
that Company may elect to pay Employee in lieu of such notice period.

3.4 Termination for Cause
Company may terminate Employee immediately for cause, including:
(a) Material breach of this Agreement
(b) Gross negligence or willful misconduct
(c) Conviction of a felony
(d) Violation of Company policies

3.5 Severance
If Company terminates Employee without cause, Employee shall receive:
(a) Continued base salary for three (3) months
(b) Pro-rated annual bonus for the year of termination
(c) Continuation of health benefits for three (3) months

3.6 Effect of Termination
Upon termination, Employee shall return all Company property and Confidential Information.

ARTICLE IV - CONFIDENTIAL INFORMATION

4.1 Definition
"Confidential Information" includes all non-public information relating to Company's business, 
including technical data, trade secrets, customer lists, and business plans.

4.2 Non-Disclosure
Employee agrees not to disclose Confidential Information during or after employment, except 
as required to perform her duties.

4.3 Return of Information
Upon termination, Employee shall return all Confidential Information and Company property.

ARTICLE V - INTELLECTUAL PROPERTY

5.1 Work Product
All work product, inventions, and developments created by Employee in the course of employment 
shall be the exclusive property of Company.

5.2 Assignment
Employee hereby assigns to Company all right, title, and interest in such work product.

5.3 Assistance
Employee agrees to assist Company in obtaining patents and copyrights for such work product.

ARTICLE VI - RESTRICTIVE COVENANTS

6.1 Non-Competition
During employment and for twelve (12) months thereafter, Employee shall not engage in any 
business competitive with Company within a fifty (50) mile radius of Company's offices.

6.2 Non-Solicitation
For twelve (12) months after termination, Employee shall not:
(a) Solicit Company's employees to leave their employment
(b) Solicit Company's customers or clients
(c) Interfere with Company's business relationships

6.3 Enforceability
Employee acknowledges that these restrictions are reasonable and necessary to protect 
Company's legitimate business interests.

ARTICLE VII - GENERAL PROVISIONS

7.1 Governing Law
This Agreement shall be governed by the laws of the State of New York.

7.2 Entire Agreement
This Agreement constitutes the entire agreement and supersedes all prior agreements.

7.3 Amendment
This Agreement may be amended only by written instrument signed by both parties.

7.4 Severability
If any provision is found invalid, the remaining provisions shall remain in effect.

7.5 Arbitration
Any disputes shall be resolved by binding arbitration in New York, New York.

IN WITNESS WHEREOF, the parties have executed this Agreement.

DATAFLOW INDUSTRIES INC.                EMPLOYEE

By: _________________________          _________________________
Name: Robert Anderson                   Jennifer Williams
Title: CEO                             
Date: April 1, 2024                    Date: April 1, 2024
"""
    
    return content


def save_sample_documents():
    """Save all sample documents to files"""
    
    # Create directory if it doesn't exist
    os.makedirs("sample_documents", exist_ok=True)
    
    documents = [
        ("Service_Agreement.txt", create_sample_service_agreement()),
        ("NDA_Agreement.txt", create_sample_nda()),
        ("Employment_Contract.txt", create_sample_employment_contract()),
    ]
    
    print("Generating sample documents...")
    print("="*60)
    
    for filename, content in documents:
        filepath = os.path.join("sample_documents", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Count sections
        sections = content.count("ARTICLE")
        words = len(content.split())
        
        print(f"✅ Created: {filename}")
        print(f"   Sections: {sections}")
        print(f"   Words: {words:,}")
        print()
    
    print("="*60)
    print("Sample documents created successfully!")
    print(f"Location: {os.path.abspath('sample_documents')}")
    print()
    print("You can now upload these files in the Hierarchical RAG app.")
    print()
    print("Suggested test queries:")
    print("- 'Compare payment terms across all documents'")
    print("- 'What are the termination conditions in each agreement?'")
    print("- 'Extract all monetary amounts and their purposes'")
    print("- 'List all confidentiality obligations'")
    print("- 'What are the key differences between these contracts?'")


if __name__ == "__main__":
    save_sample_documents()
